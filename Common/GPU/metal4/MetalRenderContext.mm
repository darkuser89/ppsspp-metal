// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Common/GPU/metal4/MetalRenderContext.h"

#include <algorithm>
#include <cstring>
#include <cstdio>

#include "Common/Log.h"

namespace Metal {

RenderContext::~RenderContext() {
	// An unsubmitted buffer has no GPU work to drain. The owner must end its
	// encoders before destroying the context, even on initialization failure.
	EndInitializationBlit();
	commands_ = nil;
	initializationCommands_ = nil;
	std::string error;
	WaitUntilIdle(&error);
}

bool RenderContext::Init(std::string *error, size_t inflightFrames) {
	error->clear();
	if (device_ || inflightFrames < 1 || inflightFrames > submitted_.size()) {
		*error = "Invalid Metal context or in-flight frame count";
		return false;
	}
	if (@available(macOS 26.0, *)) {
		device_ = MTLCreateSystemDefaultDevice();
		if (!device_ || ![device_ supportsFamily:MTLGPUFamilyMetal4]) {
			*error = "This Mac does not support Metal 4";
			return false;
		}
		NSError *nativeError = nil;
		MTL4CommandQueueDescriptor *queueDesc = [[MTL4CommandQueueDescriptor alloc] init];
		queueDesc.label = @"PPSSPP Metal 4";
		queue_ = [device_ newMTL4CommandQueueWithDescriptor:queueDesc error:&nativeError];
		MTL4CompilerDescriptor *compilerDesc = [[MTL4CompilerDescriptor alloc] init];
		compilerDesc.label = @"PPSSPP Metal 4 compiler";
		compiler_ = [device_ newCompilerWithDescriptor:compilerDesc error:&nativeError];
		if (!queue_ || !compiler_) {
			*error = nativeError ? nativeError.localizedDescription.UTF8String : "Failed to initialize Metal 4 queue or compiler";
			return false;
		}
		functionLibraries_ = [NSMapTable weakToStrongObjectsMapTable];
		inflightFrames_ = inflightFrames;
		for (size_t slot = 0; slot < inflightFrames_; ++slot) {
			allocators_[slot] = [device_ newCommandAllocator];
			initializationAllocators_[slot] = [device_ newCommandAllocator];
			MTLResidencySetDescriptor *desc = [[MTLResidencySetDescriptor alloc] init];
			desc.label = @"PPSSPP submission resources";
			desc.initialCapacity = 256;
			residency_[slot] = [device_ newResidencySetWithDescriptor:desc error:&nativeError];
			retained_[slot] = [[NSMutableSet alloc] init];
			if (!allocators_[slot] || !initializationAllocators_[slot] || !residency_[slot]) {
				*error = nativeError ? nativeError.localizedDescription.UTF8String : "Failed to allocate Metal 4 submission storage";
				return false;
			}
		}
		return true;
	}
	*error = "Metal 4 requires macOS 26 or later";
	return false;
}

std::string RenderContext::DeviceName() const {
	return device_ ? device_.name.UTF8String : "";
}

bool RenderContext::WaitForCommands(const std::shared_ptr<Submission> &completion, std::string *error) {
	if (!completion) {
		return true;
	}
	std::unique_lock<std::mutex> lock(completion->mutex);
	completion->ready.wait(lock, [&]() { return completion->complete; });
	if (!completion->error.empty()) {
		*error = completion->error;
		return false;
	}
	return true;
}

bool RenderContext::WaitForSubmission(size_t slot, std::string *error) {
	const auto completion = submitted_[slot];
	const bool success = WaitForCommands(completion, error);
	if (success && completion && completion->serial > lastCompletedSerial_) {
		const double start = completion->gpuStart;
		const double end = completion->gpuEnd;
		lastSubmissionGPUTimeMs_.store(start > 0.0 && end >= start ? (end - start) * 1000.0 : -1.0, std::memory_order_relaxed);
		lastCompletedSerial_ = completion->serial;
	}
	if (success && !submittedProfilePasses_[slot].empty()) {
		const size_t count = submittedProfileSamplesUsed_[slot];
		NSData *data = [profileBuffers_[slot] resolveCounterRange:NSMakeRange(0, count)];
		if (data.length >= count * sizeof(MTL4TimestampHeapEntry)) {
			const auto *results = (const MTL4TimestampHeapEntry *)data.bytes;
			std::string summary = "Metal 4 GPU passes (last completed submission)\n";
			char line[192];
			for (const auto &pass : submittedProfilePasses_[slot]) {
				const size_t index = pass.firstSample;
				const uint64_t start = results[index].timestamp;
				const uint64_t end = results[index + (pass.kind == ProfileKind::RENDER ? 3 : 1)].timestamp;
				if (start == 0 || end == 0 || end < start) {
					continue;
				}
				if (pass.kind != ProfileKind::RENDER) {
					snprintf(line, sizeof(line), "%s: %.3f ms\n", pass.tag.c_str(), (end - start) * 1e-6);
					summary += line;
					continue;
				}
				const uint64_t vertexStart = start;
				const uint64_t vertexEnd = results[index + 1].timestamp;
				const uint64_t fragmentStart = results[index + 2].timestamp;
				const uint64_t fragmentEnd = end;
				if (vertexStart == 0 || vertexEnd == 0 ||
					fragmentStart == 0 || fragmentEnd == 0 ||
					vertexEnd < vertexStart || fragmentEnd < fragmentStart || fragmentEnd < vertexStart) {
					continue;
				}
				snprintf(line, sizeof(line), "%s: %.3f ms (vertex %.3f, fragment %.3f)\n",
					pass.tag.c_str(), (fragmentEnd - vertexStart) * 1e-6,
					(vertexEnd - vertexStart) * 1e-6, (fragmentEnd - fragmentStart) * 1e-6);
				summary += line;
			}
			gpuProfileString_ = std::move(summary);
			if (submittedProfileLog_[slot]) {
				INFO_LOG(Log::G3D, "%s", gpuProfileString_.c_str());
			}
		} else {
			gpuProfileString_ = "Metal GPU counter results unavailable";
		}
	}
	submitted_[slot].reset();
	if (completion) {
		[residency_[slot] removeAllAllocations];
		[residency_[slot] commit];
		[retained_[slot] removeAllObjects];
	}
	submittedProfilePasses_[slot].clear();
	submittedProfileSamplesUsed_[slot] = 0;
	submittedProfileLog_[slot] = false;
	return success;
}

NSUInteger RenderContext::ReserveProfileSamples(size_t count, const char *tag, ProfileKind kind) {
	if (!profilingEnabled_ || !commands_ || profileSamplesUsed_ + count > MAX_PROFILE_SAMPLES) {
		return NSNotFound;
	}
	const size_t slot = nextSubmission_;
	if (!profileBuffers_[slot]) {
		MTL4CounterHeapDescriptor *desc = [[MTL4CounterHeapDescriptor alloc] init];
		desc.type = MTL4CounterHeapTypeTimestamp;
		desc.count = MAX_PROFILE_SAMPLES;
		profileBuffers_[slot] = [device_ newCounterHeapWithDescriptor:desc error:nil];
		if (!profileBuffers_[slot]) {
			gpuProfileString_ = "Metal GPU timestamp counters unavailable";
			return NSNotFound;
		}
	}
	const NSUInteger index = profileSamplesUsed_;
	profileSamplesUsed_ += count;
	profilePasses_.push_back({tag ? tag : "GPU pass", index, kind});
	return index;
}

void RenderContext::Keep(id object) {
	if (!object || [retained_[nextSubmission_] containsObject:object]) {
		return;
	}
	[retained_[nextSubmission_] addObject:object];
	if ([object conformsToProtocol:@protocol(MTLAllocation)]) {
		[residency_[nextSubmission_] addAllocation:object];
	}
	if ([object conformsToProtocol:@protocol(MTLTexture)]) {
		id<MTLTexture> texture = object;
		Keep(texture.parentTexture);
		Keep(texture.buffer);
	}
}

void RenderContext::ProfileRenderPass(MTL4RenderPassDescriptor *pass, const char *tag) {
	nextProfileTag_ = tag ? tag : "Render";
}

void RenderContext::PrepareEncoder(id<MTL4CommandEncoder> encoder, const char *tag, ProfileKind kind) {
	if (!encoder) {
		return;
	}
	if (tag) {
		encoder.label = [NSString stringWithUTF8String:tag];
	}
	// PSP render targets are repeatedly sampled, copied and rebound. Preserve
	// visibility across passes and submissions, including initialization uploads.
	[encoder barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
	auto &state = encoderStates_[(__bridge void *)encoder];
	state.kind = kind;
	state.profileIndex = ReserveProfileSamples(kind == ProfileKind::RENDER ? 4 : 2, tag, kind);
	const NSUInteger index = state.profileIndex;
	if (index != NSNotFound) {
		if (kind == ProfileKind::RENDER) {
			id<MTL4RenderCommandEncoder> render = (id<MTL4RenderCommandEncoder>)encoder;
			[render writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageVertex intoHeap:profileBuffers_[nextSubmission_] atIndex:index];
			[render writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageFragment intoHeap:profileBuffers_[nextSubmission_] atIndex:index + 2];
		} else {
			[(id<MTL4ComputeCommandEncoder>)encoder writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed intoHeap:profileBuffers_[nextSubmission_] atIndex:index];
		}
	}
}

id<MTL4RenderCommandEncoder> RenderContext::RenderEncoder(id<MTL4CommandBuffer> commands, MTL4RenderPassDescriptor *pass, const char *tag) {
	for (NSUInteger i = 0; i < 8; ++i) {
		Keep({pass.colorAttachments[i].texture, pass.colorAttachments[i].resolveTexture});
	}
	Keep({pass.depthAttachment.texture, pass.depthAttachment.resolveTexture,
		pass.stencilAttachment.texture, pass.stencilAttachment.resolveTexture});
	id<MTL4RenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
	PrepareEncoder(encoder, tag ? tag : nextProfileTag_.c_str(), ProfileKind::RENDER);
	nextProfileTag_.clear();
	return encoder;
}

id<MTL4ComputeCommandEncoder> RenderContext::BlitEncoder(id<MTL4CommandBuffer> commands, const char *tag) {
	id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
	PrepareEncoder(encoder, tag, ProfileKind::BLIT);
	return encoder;
}

id<MTL4ComputeCommandEncoder> RenderContext::ComputeEncoder(id<MTL4CommandBuffer> commands, const char *tag) {
	id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
	PrepareEncoder(encoder, tag, ProfileKind::COMPUTE);
	return encoder;
}

void RenderContext::EndEncoder(id<MTL4CommandEncoder> encoder) {
	if (!encoder) {
		return;
	}
	auto found = encoderStates_.find((__bridge void *)encoder);
	if (found != encoderStates_.end()) {
		const auto &state = found->second;
		const NSUInteger index = state.profileIndex;
		if (index != NSNotFound) {
			if (state.kind == ProfileKind::RENDER) {
				id<MTL4RenderCommandEncoder> render = (id<MTL4RenderCommandEncoder>)encoder;
				[render writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageVertex intoHeap:profileBuffers_[nextSubmission_] atIndex:index + 1];
				[render writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageFragment intoHeap:profileBuffers_[nextSubmission_] atIndex:index + 3];
			} else {
				[(id<MTL4ComputeCommandEncoder>)encoder writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed intoHeap:profileBuffers_[nextSubmission_] atIndex:index + 1];
			}
		}
		encoderStates_.erase(found);
	}
	[encoder endEncoding];
}

id<MTL4ArgumentTable> RenderContext::NewArgumentTable() {
	auto &pool = argumentPools_[nextSubmission_];
	if (argumentPoolUsed_ == pool.size()) {
		MTL4ArgumentTableDescriptor *desc = [[MTL4ArgumentTableDescriptor alloc] init];
		desc.maxBufferBindCount = 31;
		desc.maxTextureBindCount = 16;
		desc.maxSamplerStateBindCount = 16;
		desc.initializeBindings = YES;
		NSError *error = nil;
		id<MTL4ArgumentTable> table = [device_ newArgumentTableWithDescriptor:desc error:&error];
		if (!table) {
			ERROR_LOG(Log::G3D, "Metal 4 argument table: %s", error.localizedDescription.UTF8String);
			return nil;
		}
		pool.push_back(table);
	}
	id<MTL4ArgumentTable> table = pool[argumentPoolUsed_++];
	for (NSUInteger i = 0; i < 31; ++i) {
		[table setAddress:0 atIndex:i];
	}
	for (NSUInteger i = 0; i < 16; ++i) {
		[table setTexture:(MTLResourceID){0} atIndex:i];
		[table setSamplerState:(MTLResourceID){0} atIndex:i];
	}
	return table;
}

id<MTL4ArgumentTable> RenderContext::Arguments(id<MTL4CommandEncoder> encoder, MTLRenderStages stage) {
	auto &state = encoderStates_.at((__bridge void *)encoder);
	auto &table = stage == MTLRenderStageFragment ? state.fragment : state.vertex;
	if (!table) {
		table = NewArgumentTable();
		if (state.kind == ProfileKind::RENDER) {
			[(id<MTL4RenderCommandEncoder>)encoder setArgumentTable:table atStages:stage];
		} else {
			[(id<MTL4ComputeCommandEncoder>)encoder setArgumentTable:table];
		}
	}
	return table;
}

void RenderContext::SetBuffer(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, id<MTLBuffer> buffer, size_t offset, uint32_t index) {
	Keep(buffer);
	[Arguments(encoder, stage) setAddress:buffer ? buffer.gpuAddress + offset : 0 atIndex:index];
}

void RenderContext::SetBytes(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, const void *bytes, size_t size, uint32_t index) {
	std::string error;
	const auto upload = Upload(bytes, size, &error);
	SetBuffer(encoder, stage, upload.buffer, upload.offset, index);
}

void RenderContext::SetTexture(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, id<MTLTexture> texture, uint32_t index) {
	Keep(texture);
	[Arguments(encoder, stage) setTexture:texture ? texture.gpuResourceID : (MTLResourceID){0} atIndex:index];
}

void RenderContext::SetSampler(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, id<MTLSamplerState> sampler, uint32_t index) {
	Keep(sampler);
	[Arguments(encoder, stage) setSamplerState:sampler ? sampler.gpuResourceID : (MTLResourceID){0} atIndex:index];
}

void RenderContext::SetBuffer(id<MTL4ComputeCommandEncoder> encoder, id<MTLBuffer> buffer, size_t offset, uint32_t index) {
	Keep(buffer);
	[Arguments(encoder, MTLRenderStageVertex) setAddress:buffer ? buffer.gpuAddress + offset : 0 atIndex:index];
}

void RenderContext::SetBytes(id<MTL4ComputeCommandEncoder> encoder, const void *bytes, size_t size, uint32_t index) {
	std::string error;
	const auto upload = Upload(bytes, size, &error);
	SetBuffer(encoder, upload.buffer, upload.offset, index);
}

void RenderContext::SetTexture(id<MTL4ComputeCommandEncoder> encoder, id<MTLTexture> texture, uint32_t index) {
	Keep(texture);
	[Arguments(encoder, MTLRenderStageVertex) setTexture:texture ? texture.gpuResourceID : (MTLResourceID){0} atIndex:index];
}

void RenderContext::SetSampler(id<MTL4ComputeCommandEncoder> encoder, id<MTLSamplerState> sampler, uint32_t index) {
	Keep(sampler);
	[Arguments(encoder, MTLRenderStageVertex) setSamplerState:sampler ? sampler.gpuResourceID : (MTLResourceID){0} atIndex:index];
}

id<MTL4CommandBuffer> RenderContext::InitializationCommands(std::string *error) {
	error->clear();
	if (!commands_) {
		*error = "Metal resource initialization requires an active command buffer";
		return nil;
	}
	if (!initializationCommands_) {
		initializationCommands_ = [device_ newCommandBuffer];
		if (!initializationCommands_) {
			*error = "Failed to allocate Metal resource initialization commands";
			return nil;
		}
		[initializationCommands_ beginCommandBufferWithAllocator:initializationAllocators_[nextSubmission_]];
		[initializationCommands_ useResidencySet:residency_[nextSubmission_]];
		Keep(initializationCommands_);
		initializationCommands_.label = @"PPSSPP resource initialization";
	}
	EndInitializationBlit();
	return initializationCommands_;
}

void RenderContext::EndInitializationBlit() {
	if (initializationBlitEncoder_) {
		EndEncoder(initializationBlitEncoder_);
		initializationBlitEncoder_ = nil;
	}
}

id<MTL4ComputeCommandEncoder> RenderContext::InitializationBlitEncoder(std::string *error) {
	if (!initializationBlitEncoder_) {
		auto commands = InitializationCommands(error);
		if (!commands) {
			return nil;
		}
		initializationBlitEncoder_ = BlitEncoder(commands, "Texture initialization blit");
		if (!initializationBlitEncoder_) {
			*error = "Failed to encode Metal texture initialization";
		}
	}
	return initializationBlitEncoder_;
}

bool RenderContext::BeginCommands(std::string *error) {
	error->clear();
	if (!queue_ || commands_) {
		*error = "Metal command queue is unavailable or a command buffer is already active";
		return false;
	}
	// Limit outstanding work, and surface asynchronous execution errors when
	// recycling the slot. Always clear the slot, including the failure path.
	bool success = WaitForSubmission(nextSubmission_, error);
	if (!success) {
		return false;
	}
	// Reuse command storage and upload memory only after GPU completion.
	[allocators_[nextSubmission_] reset];
	[initializationAllocators_[nextSubmission_] reset];
	[profileBuffers_[nextSubmission_] invalidateCounterRange:NSMakeRange(0, MAX_PROFILE_SAMPLES)];
	argumentPoolUsed_ = 0;
	auto &blocks = uploadBlocks_[nextSubmission_];
	while (!blocks.empty() && blocks.back().used == 0) {
		blocks.pop_back();
	}
	for (auto &block : blocks) {
		block.used = 0;
	}
	currentUploadBlock_ = 0;
	commands_ = [device_ newCommandBuffer];
	if (!commands_) {
		*error = "Failed to allocate a Metal command buffer";
		return false;
	}
	[commands_ beginCommandBufferWithAllocator:allocators_[nextSubmission_]];
	[commands_ useResidencySet:residency_[nextSubmission_]];
	commands_.label = @"PPSSPP render commands";
	Keep(commands_);
	currentCompletion_ = std::make_shared<Submission>();
	++commandGeneration_;
	if (beginCommandsCallback_) {
		beginCommandsCallback_();
	}
	return true;
}

UploadSlice RenderContext::ReserveUpload(size_t size, std::string *error) {
	error->clear();
	if (!device_ || !size || size > device_.maxBufferLength) {
		*error = "Invalid Metal transient upload size or device";
		return {};
	}
	if (!commands_) {
		*error = "Metal transient uploads require an active command buffer";
		return {};
	}
	auto &blocks = uploadBlocks_[nextSubmission_];
	while (currentUploadBlock_ < blocks.size()) {
		auto &block = blocks[currentUploadBlock_];
		// Align vertex, index and uniform addresses.
		const size_t offset = (block.used + 255) & ~size_t(255);
		if (offset <= block.buffer.length && size <= block.buffer.length - offset) {
			block.used = offset + size;
			Keep(block.buffer);
			return {block.buffer, offset};
		}
		++currentUploadBlock_;
	}
	const size_t capacity = std::max<size_t>(1024 * 1024, size);
	id<MTLBuffer> buffer = [device_ newBufferWithLength:capacity options:MTLResourceStorageModeShared];
	if (!buffer) {
		*error = "Failed to allocate Metal transient upload block";
		return {};
	}
	buffer.label = @"PPSSPP transient uploads";
	blocks.push_back({buffer, size});
	Keep(buffer);
	return {buffer, 0};
}

UploadSlice RenderContext::Upload(const void *data, size_t size, std::string *error) {
	if (!data) {
		*error = "Invalid Metal transient upload data";
		return {};
	}
	auto slice = ReserveUpload(size, error);
	if (slice) {
		memcpy((uint8_t *)slice.buffer.contents + slice.offset, data, size);
	}
	return slice;
}

id<MTLBuffer> RenderContext::AcquireReadbackBuffer(size_t size) {
	if (!device_ || !size || size > device_.maxBufferLength) {
		return nil;
	}
	if (readbackBuffer_ && readbackBuffer_.length >= size) {
		id<MTLBuffer> buffer = readbackBuffer_;
		readbackBuffer_ = nil;
		Keep(buffer);
		return buffer;
	}
	id<MTLBuffer> buffer = [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
	Keep(buffer);
	return buffer;
}

void RenderContext::RecycleReadbackBuffer(id<MTLBuffer> buffer) {
	constexpr size_t MAX_CACHED_READBACK = 1024 * 1024;
	if (buffer && buffer.length <= MAX_CACHED_READBACK &&
		(!readbackBuffer_ || readbackBuffer_.length < buffer.length)) {
		readbackBuffer_ = buffer;
	}
}

void RenderContext::PresentDrawable(id<MTLDrawable> drawable) {
	pendingDrawable_ = drawable;
	Keep(drawable);
}

bool RenderContext::SubmitCommands(bool wait, std::string *error) {
	error->clear();
	if (!commands_) {
		*error = "No active Metal command buffer to submit";
		return false;
	}
	EndInitializationBlit();
	const size_t slot = nextSubmission_;
	[residency_[slot] commit];
	id<MTL4CommandBuffer> buffers[2];
	NSUInteger count = 0;
	if (initializationCommands_) {
		[initializationCommands_ endCommandBuffer];
		buffers[count++] = initializationCommands_;
	}
	[commands_ endCommandBuffer];
	buffers[count++] = commands_;
	const auto completion = currentCompletion_;
	completion->serial = ++nextSubmissionSerial_;
	MTL4CommitOptions *options = [[MTL4CommitOptions alloc] init];
	[options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
		{
			std::lock_guard<std::mutex> lock(completion->mutex);
			completion->gpuStart = feedback.GPUStartTime;
			completion->gpuEnd = feedback.GPUEndTime;
			if (feedback.error) {
				completion->error = feedback.error.localizedDescription.UTF8String;
			}
			completion->complete = true;
		}
		completion->ready.notify_all();
	}];
	if (pendingDrawable_) {
		[queue_ waitForDrawable:pendingDrawable_];
	}
	[queue_ commit:buffers count:count options:options];
	if (pendingDrawable_) {
		[queue_ signalDrawable:pendingDrawable_];
		[pendingDrawable_ present];
		pendingDrawable_ = nil;
	}
	submitted_[slot] = completion;
	submittedProfilePasses_[slot] = std::move(profilePasses_);
	submittedProfileSamplesUsed_[slot] = profileSamplesUsed_;
	submittedProfileLog_[slot] = profileLogEnabled_;
	profilePasses_.clear();
	profileSamplesUsed_ = 0;
	commands_ = nil;
	initializationCommands_ = nil;
	currentCompletion_.reset();
	nextSubmission_ = (nextSubmission_ + 1) % inflightFrames_;
	return !wait || WaitForSubmission(slot, error);
}

bool RenderContext::WaitUntilIdle(std::string *error) {
	error->clear();
	bool success = true;
	for (size_t slot = 0; slot < submitted_.size(); ++slot) {
		std::string commandError;
		if (!WaitForSubmission(slot, &commandError)) {
			if (success) {
				*error = commandError;
			}
			success = false;
		}
	}
	return success;
}

MTL4LibraryFunctionDescriptor *RenderContext::FunctionDescriptor(id<MTLFunction> function) const {
	MTL4LibraryFunctionDescriptor *desc = [[MTL4LibraryFunctionDescriptor alloc] init];
	desc.name = function.name;
	@synchronized (functionLibraries_) {
		desc.library = [functionLibraries_ objectForKey:function];
	}
	return desc;
}

id<MTLComputePipelineState> RenderContext::CreateComputePipeline(id<MTLFunction> function, NSError **error) {
	MTL4ComputePipelineDescriptor *desc = [[MTL4ComputePipelineDescriptor alloc] init];
	desc.computeFunctionDescriptor = FunctionDescriptor(function);
	return [compiler_ newComputePipelineStateWithDescriptor:desc compilerTaskOptions:nil error:error];
}

id<MTLFunction> RenderContext::CreateShader(const CompiledShader &shader, const char *tag, std::string *error) {
	error->clear();
	if (!compiler_ || shader.source.empty() || shader.entryPoint.empty()) {
		*error = "Metal shader or compiler is unavailable";
		return nil;
	}
	@autoreleasepool {
		MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
		options.languageVersion = MTLLanguageVersion4_0;
		// Preserve NaN/Inf range culling and coplanar position invariance.
		options.mathMode = MTLMathModeRelaxed;
		options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
		options.preserveInvariance = YES;
		MTL4LibraryDescriptor *desc = [[MTL4LibraryDescriptor alloc] init];
		desc.source = [[NSString alloc] initWithBytes:shader.source.data() length:shader.source.size() encoding:NSUTF8StringEncoding];
		desc.options = options;
		desc.name = tag ? [NSString stringWithUTF8String:tag] : @"PPSSPP";
		NSString *entryPoint = [[NSString alloc] initWithBytes:shader.entryPoint.data() length:shader.entryPoint.size() encoding:NSUTF8StringEncoding];
		if (!desc.source || !entryPoint) {
			*error = "Metal shader source or entry point is not valid UTF-8";
			return nil;
		}
		NSError *compileError = nil;
		id<MTLLibrary> library = [compiler_ newLibraryWithDescriptor:desc error:&compileError];
		if (!library) {
			*error = compileError ? compileError.localizedDescription.UTF8String : "Metal 4 library compilation failed";
			return nil;
		}
		id<MTLFunction> function = [library newFunctionWithName:entryPoint];
		if (!function) {
			*error = "Metal shader entry point was not found: " + shader.entryPoint;
		} else {
			@synchronized (functionLibraries_) {
				[functionLibraries_ setObject:library forKey:function];
			}
		}
		return function;
	}
}

}  // namespace Metal
