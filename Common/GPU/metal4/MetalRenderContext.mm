// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Common/GPU/metal4/MetalRenderContext.h"

#include <algorithm>
#include <cstring>
#include <cstdio>

#include "Common/Log.h"
#include "Common/TimeUtil.h"

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
		timestampFrequency_ = [device_ queryTimestampFrequency];
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
	const double start = time_now_d();
	++metrics_.completionWaits;
	std::unique_lock<std::mutex> lock(completion->mutex);
	if (!completion->complete) {
		++metrics_.blockingCompletionWaits;
	}
	completion->ready.wait(lock, [&]() { return completion->complete; });
	metrics_.completionWaitSeconds += time_now_d() - start;
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
			std::string summary = "Metal 4 GPU passes (last completed submission)\nDependency counts cover declared accesses within this submission, not GPU wait durations or previous submissions.\n";
			dependencyProfile_ = {};
			dependencyProfile_.incomplete = submittedProfileIncomplete_[slot];
			if (dependencyProfile_.incomplete) {
				summary += "Dependency trace incomplete: timestamp capacity or external encoder work.\n";
			}
			ResourceAccesses initializationAccesses;
			ResourceAccesses mainAccesses;
			const double tickMs = timestampFrequency_ ? 1000.0 / timestampFrequency_ : 0.0;
			char line[192];
			for (const auto &pass : submittedProfilePasses_[slot]) {
				auto dependencies = FindDependencies(initializationAccesses, pass.accesses);
				const uint64_t initializationHazards = dependencies.raw + dependencies.war + dependencies.waw;
				if (!pass.initialization) {
					dependencies.Include(FindDependencies(mainAccesses, pass.accesses));
					dependencyProfile_.initializationToMainHazards += initializationHazards;
				} else {
					++dependencyProfile_.initializationPasses;
				}
				++dependencyProfile_.passes;
				dependencyProfile_.resourceAccesses += pass.accesses.size();
				dependencyProfile_.hazards.Include(dependencies);
				MergeAccesses(pass.initialization ? &initializationAccesses : &mainAccesses, pass.accesses);
				snprintf(line, sizeof(line), "Dependencies %s resources=%zu RAW/WAR/WAW=%llu/%llu/%llu stages=%llx->%llx init=%llu\n",
					pass.initialization ? "initialization" : "main", pass.accesses.size(),
					(unsigned long long)dependencies.raw, (unsigned long long)dependencies.war, (unsigned long long)dependencies.waw,
					(unsigned long long)dependencies.afterStages, (unsigned long long)dependencies.beforeStages,
					(unsigned long long)initializationHazards);
				summary += line;
				const size_t index = pass.firstSample;
				if (pass.kind == ProfileKind::BLIT) {
					summary += pass.tag + ": blit duration unavailable from encoder timestamps\n";
					continue;
				}
				const uint64_t start = results[index].timestamp;
				const uint64_t end = results[index + (pass.kind == ProfileKind::RENDER ? 3 : 1)].timestamp;
				if (!timestampFrequency_ || start == 0 || end == 0 || end < start) {
					continue;
				}
				if (pass.kind != ProfileKind::RENDER) {
					snprintf(line, sizeof(line), "%s: %.3f ms\n", pass.tag.c_str(), (end - start) * tickMs);
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
					pass.tag.c_str(), (fragmentEnd - vertexStart) * tickMs,
					(vertexEnd - vertexStart) * tickMs, (fragmentEnd - fragmentStart) * tickMs);
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
	submittedAccesses_[slot].clear();
	submittedUnknownAccesses_[slot] = false;
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
	if (!profilingEnabled_ || !commands_) {
		return NSNotFound;
	}
	if (profileSamplesUsed_ + count > MAX_PROFILE_SAMPLES) {
		profileIncomplete_ = true;
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
			profileIncomplete_ = true;
			return NSNotFound;
		}
	}
	const NSUInteger index = profileSamplesUsed_;
	profileSamplesUsed_ += count;
	profilePasses_.push_back({tag ? tag : "GPU pass", index, kind, false, {}});
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

void RenderContext::PrepareEncoder(id<MTL4CommandEncoder> encoder, const char *tag, ProfileKind kind, bool declaredAccesses) {
	if (!encoder) {
		return;
	}
	if (tag) {
		encoder.label = [NSString stringWithUTF8String:tag];
	}
	auto &state = encoderStates_[(__bridge void *)encoder];
	state.kind = kind;
	state.initialization = encoder.commandBuffer == initializationCommands_;
	if (!declaredAccesses) {
		(state.initialization ? initializationUnknownAccesses_ : mainUnknownAccesses_) = true;
	}
	if (kind == ProfileKind::RENDER) {
		++metrics_.renderPasses;
	} else if (kind == ProfileKind::BLIT) {
		++metrics_.blitPasses;
	} else {
		++metrics_.computePasses;
	}
	state.profileIndex = ReserveProfileSamples(kind == ProfileKind::RENDER ? 4 : 2, tag, kind);
	const NSUInteger index = state.profileIndex;
	if (index != NSNotFound) {
		state.profilePass = profilePasses_.size() - 1;
		profilePasses_.back().initialization = encoder.commandBuffer == initializationCommands_;
		if (kind == ProfileKind::RENDER) {
			id<MTL4RenderCommandEncoder> render = (id<MTL4RenderCommandEncoder>)encoder;
			[render writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageVertex intoHeap:profileBuffers_[nextSubmission_] atIndex:index];
			[render writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed afterStage:MTLRenderStageFragment intoHeap:profileBuffers_[nextSubmission_] atIndex:index + 2];
		} else {
			[(id<MTL4ComputeCommandEncoder>)encoder writeTimestampWithGranularity:MTL4TimestampGranularityRelaxed intoHeap:profileBuffers_[nextSubmission_] atIndex:index];
		}
	}
	if (conservativeBarriers_ || !declaredAccesses) {
		DependencyBarrier(encoder, MTLStageAll, MTLStageAll);
	}
}

id<MTL4RenderCommandEncoder> RenderContext::RenderEncoder(id<MTL4CommandBuffer> commands, MTL4RenderPassDescriptor *pass, const char *tag) {
	for (NSUInteger i = 0; i < 8; ++i) {
		Keep({pass.colorAttachments[i].texture, pass.colorAttachments[i].resolveTexture});
	}
	Keep({pass.depthAttachment.texture, pass.depthAttachment.resolveTexture,
		pass.stencilAttachment.texture, pass.stencilAttachment.resolveTexture});
	id<MTL4RenderCommandEncoder> encoder = [commands renderCommandEncoderWithDescriptor:pass];
	PrepareEncoder(encoder, tag ? tag : nextProfileTag_.c_str(), ProfileKind::RENDER, true);
	for (NSUInteger i = 0; i < 8; ++i) {
		UseResource(encoder, pass.colorAttachments[i].texture, MTLStageFragment, MTLStageFragment);
		UseResource(encoder, pass.colorAttachments[i].resolveTexture, (MTLStages)0, MTLStageFragment);
	}
	UseResource(encoder, pass.depthAttachment.texture, MTLStageFragment, MTLStageFragment);
	UseResource(encoder, pass.stencilAttachment.texture, MTLStageFragment, MTLStageFragment);
	UseResource(encoder, pass.depthAttachment.resolveTexture, (MTLStages)0, MTLStageFragment);
	UseResource(encoder, pass.stencilAttachment.resolveTexture, (MTLStages)0, MTLStageFragment);
	nextProfileTag_.clear();
	return encoder;
}

id<MTL4ComputeCommandEncoder> RenderContext::BlitEncoder(id<MTL4CommandBuffer> commands, const char *tag, bool declaredAccesses) {
	id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
	PrepareEncoder(encoder, tag, ProfileKind::BLIT, declaredAccesses);
	return encoder;
}

id<MTL4ComputeCommandEncoder> RenderContext::ComputeEncoder(id<MTL4CommandBuffer> commands, const char *tag, bool declaredAccesses) {
	id<MTL4ComputeCommandEncoder> encoder = [commands computeCommandEncoder];
	PrepareEncoder(encoder, tag, ProfileKind::COMPUTE, declaredAccesses);
	return encoder;
}

void RenderContext::EndEncoder(id<MTL4CommandEncoder> encoder) {
	if (!encoder) {
		return;
	}
	auto found = encoderStates_.find((__bridge void *)encoder);
	if (found != encoderStates_.end()) {
		const auto &state = found->second;
		MergeAccesses(state.initialization ? &initializationAccesses_ : &mainAccesses_, state.accesses);
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

RenderContext::ArgumentTable *RenderContext::NewArgumentTable() {
	auto &pool = argumentPools_[nextSubmission_];
	const bool fresh = argumentPoolUsed_ == pool.size();
	if (fresh) {
		MTL4ArgumentTableDescriptor *desc = [[MTL4ArgumentTableDescriptor alloc] init];
		desc.maxBufferBindCount = 31;
		desc.maxTextureBindCount = 16;
		desc.maxSamplerStateBindCount = 16;
		desc.initializeBindings = YES;
		NSError *error = nil;
		id<MTL4ArgumentTable> table = [device_ newArgumentTableWithDescriptor:desc error:&error];
		if (!table) {
			ERROR_LOG(Log::G3D, "Metal 4 argument table: %s", error.localizedDescription.UTF8String);
			return nullptr;
		}
		pool.push_back({table});
		++metrics_.argumentAllocations;
	}
	auto &entry = pool[argumentPoolUsed_++];
	id<MTL4ArgumentTable> table = entry.table;
	argumentUsage_[nextSubmission_] = argumentPoolUsed_;
	if (!fresh) {
		for (NSUInteger i = 0; i < 31; ++i) {
			if (entry.buffers & (1u << i)) {
				[table setAddress:0 atIndex:i];
				++metrics_.argumentBindingsCleared;
			}
		}
		for (NSUInteger i = 0; i < 16; ++i) {
			if (entry.textures & (1u << i)) {
				[table setTexture:(MTLResourceID){0} atIndex:i];
				++metrics_.argumentBindingsCleared;
			}
			if (entry.samplers & (1u << i)) {
				[table setSamplerState:(MTLResourceID){0} atIndex:i];
				++metrics_.argumentBindingsCleared;
			}
		}
	}
	entry.buffers = entry.textures = entry.samplers = 0;
	return &entry;
}

RenderContext::ArgumentTable *RenderContext::Arguments(id<MTL4CommandEncoder> encoder, MTLRenderStages stage) {
	auto &state = encoderStates_.at((__bridge void *)encoder);
	auto &index = stage == MTLRenderStageFragment ? state.fragment : state.vertex;
	if (index == SIZE_MAX) {
		const size_t next = argumentPoolUsed_;
		auto entry = NewArgumentTable();
		if (!entry) {
			return nullptr;
		}
		index = next;
		if (state.kind == ProfileKind::RENDER) {
			[(id<MTL4RenderCommandEncoder>)encoder setArgumentTable:entry->table atStages:stage];
		} else {
			[(id<MTL4ComputeCommandEncoder>)encoder setArgumentTable:entry->table];
		}
	}
	return &argumentPools_[nextSubmission_][index];
}

void RenderContext::SetBuffer(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, id<MTLBuffer> buffer, size_t offset, uint32_t index) {
	UseResource(encoder, buffer, (MTLStages)stage, (MTLStages)0);
	if (auto entry = Arguments(encoder, stage)) {
		[entry->table setAddress:buffer ? buffer.gpuAddress + offset : 0 atIndex:index];
		entry->buffers |= 1u << index;
	}
}

void RenderContext::SetBytes(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, const void *bytes, size_t size, uint32_t index) {
	std::string error;
	const auto upload = Upload(bytes, size, &error);
	SetBuffer(encoder, stage, upload.buffer, upload.offset, index);
}

bool RenderContext::BindViewMask(id<MTL4RenderCommandEncoder> encoder, std::string *error) {
	if (!viewMask_) {
		const uint32_t mask[] = { 0, 2 };
		viewMask_ = Upload(mask, sizeof(mask), error);
		if (!viewMask_) {
			return false;
		}
	}
	SetBuffer(encoder, MTLRenderStageVertex, viewMask_.buffer, viewMask_.offset, VIEW_MASK_BUFFER_SLOT);
	SetBuffer(encoder, MTLRenderStageFragment, viewMask_.buffer, viewMask_.offset, VIEW_MASK_BUFFER_SLOT);
	return true;
}

void RenderContext::SetTexture(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, id<MTLTexture> texture, uint32_t index) {
	UseResource(encoder, texture, (MTLStages)stage, (MTLStages)0);
	if (auto entry = Arguments(encoder, stage)) {
		[entry->table setTexture:texture ? texture.gpuResourceID : (MTLResourceID){0} atIndex:index];
		entry->textures |= 1u << index;
	}
}

void RenderContext::SetSampler(id<MTL4RenderCommandEncoder> encoder, MTLRenderStages stage, id<MTLSamplerState> sampler, uint32_t index) {
	Keep(sampler);
	if (auto entry = Arguments(encoder, stage)) {
		[entry->table setSamplerState:sampler ? sampler.gpuResourceID : (MTLResourceID){0} atIndex:index];
		entry->samplers |= 1u << index;
	}
}

void RenderContext::SetBuffer(id<MTL4ComputeCommandEncoder> encoder, id<MTLBuffer> buffer, size_t offset, uint32_t index) {
	UseResource(encoder, buffer, MTLStageDispatch, (MTLStages)0);
	if (auto entry = Arguments(encoder, MTLRenderStageVertex)) {
		[entry->table setAddress:buffer ? buffer.gpuAddress + offset : 0 atIndex:index];
		entry->buffers |= 1u << index;
	}
}

void RenderContext::SetBytes(id<MTL4ComputeCommandEncoder> encoder, const void *bytes, size_t size, uint32_t index) {
	std::string error;
	const auto upload = Upload(bytes, size, &error);
	SetBuffer(encoder, upload.buffer, upload.offset, index);
}

void RenderContext::SetTexture(id<MTL4ComputeCommandEncoder> encoder, id<MTLTexture> texture, uint32_t index, MTLResourceUsage usage) {
	UseResource(encoder, texture, usage & MTLResourceUsageRead ? MTLStageDispatch : (MTLStages)0,
		usage & MTLResourceUsageWrite ? MTLStageDispatch : (MTLStages)0);
	if (auto entry = Arguments(encoder, MTLRenderStageVertex)) {
		[entry->table setTexture:texture ? texture.gpuResourceID : (MTLResourceID){0} atIndex:index];
		entry->textures |= 1u << index;
	}
}

void RenderContext::SetSampler(id<MTL4ComputeCommandEncoder> encoder, id<MTLSamplerState> sampler, uint32_t index) {
	Keep(sampler);
	if (auto entry = Arguments(encoder, MTLRenderStageVertex)) {
		[entry->table setSamplerState:sampler ? sampler.gpuResourceID : (MTLResourceID){0} atIndex:index];
		entry->samplers |= 1u << index;
	}
}

void RenderContext::DependencyBarrier(id<MTL4CommandEncoder> encoder, uint64_t after, uint64_t before) {
	if (!after || !before) {
		return;
	}
	auto &state = encoderStates_.at((__bridge void *)encoder);
	constexpr uint64_t stages[] = {MTLStageVertex, MTLStageFragment, MTLStageBlit, MTLStageDispatch};
	uint64_t missing = 0;
	uint64_t consumers = 0;
	for (size_t i = 0; i < std::size(stages); ++i) {
		if ((before & stages[i]) && (after & ~state.synchronizedStages[i])) {
			missing |= after & ~state.synchronizedStages[i];
			consumers |= stages[i];
		}
	}
	if (!missing) {
		return;
	}
	if (before == MTLStageAll) {
		consumers = MTLStageAll;
	}
	[encoder barrierAfterQueueStages:(MTLStages)missing beforeStages:(MTLStages)consumers visibilityOptions:MTL4VisibilityOptionDevice];
	++metrics_.queueBarriers;
	metrics_.fullQueueBarriers += missing == MTLStageAll && consumers == MTLStageAll;
	for (size_t i = 0; i < std::size(stages); ++i) {
		if (consumers & stages[i]) {
			state.synchronizedStages[i] |= missing;
		}
	}
}

void RenderContext::UseResource(id<MTL4CommandEncoder> encoder, id<MTLResource> resource, MTLStages reads, MTLStages writes) {
	if (!encoder || !resource) {
		return;
	}
	auto found = encoderStates_.find((__bridge void *)encoder);
	if (found == encoderStates_.end()) {
		return;
	}
	const uintptr_t object = (uintptr_t)(__bridge void *)resource;
	auto identity = resourceIdentities_.find(object);
	if (identity == resourceIdentities_.end()) {
		Keep(resource);
		if ([resource conformsToProtocol:@protocol(MTLTexture)]) {
			id<MTLTexture> texture = (id<MTLTexture>)resource;
			while (texture.parentTexture) {
				texture = texture.parentTexture;
			}
			resource = texture.buffer ? (id<MTLResource>)texture.buffer : (id<MTLResource>)texture;
		}
		identity = resourceIdentities_.emplace(object, (uintptr_t)(__bridge void *)resource).first;
		++metrics_.resourceIdentityResolutions;
	}
	const uintptr_t key = identity->second;
	auto &state = found->second;
	auto &access = state.accesses[key];
	const ResourceAccess added{reads & ~access.reads, writes & ~access.writes};
	if (!added.reads && !added.writes) {
		return;
	}
	PassDependencies dependencies;
	auto include = [&](const ResourceAccesses &history) {
		auto prior = history.find(key);
		if (prior != history.end()) {
			dependencies.Include(FindDependencies(prior->second, added));
		}
	};
	bool unknown = state.initialization ? initializationUnknownAccesses_ : mainUnknownAccesses_;
	for (size_t slot = 0; slot < submittedAccesses_.size(); ++slot) {
		include(submittedAccesses_[slot]);
		unknown |= submittedUnknownAccesses_[slot];
	}
	metrics_.priorSubmissionDependencies += dependencies.raw + dependencies.war + dependencies.waw;
	// The final initialization producer barrier orders all init commands before
	// main, even initialization recorded after this consumer was encoded.
	include(state.initialization ? initializationAccesses_ : mainAccesses_);
	metrics_.resourceDependencies += dependencies.raw + dependencies.war + dependencies.waw;
	if (unknown) {
		dependencies.afterStages = MTLStageAll;
		dependencies.beforeStages = added.reads | added.writes;
	}
	DependencyBarrier(encoder, dependencies.afterStages, dependencies.beforeStages);
	access.reads |= reads;
	access.writes |= writes;
	if (state.profilePass != NSNotFound) {
		profilePasses_[state.profilePass].accesses[key] = access;
	}
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
		initializationBlitEncoder_ = BlitEncoder(commands, "Texture initialization blit", true);
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
	// Keep() retains every cached object for this submission. Clear identities
	// before reusing storage so recycled native addresses cannot hit old entries.
	resourceIdentities_.clear();
	auto &copyPlane = copyPlanes_[nextSubmission_];
	const size_t copyTarget = copyPlane.used ? std::max<size_t>(4096, copyPlane.used * 2) : 0;
	if (copyPlane.buffer && copyPlane.buffer.length > copyTarget * 2) {
		if (++copyPlane.lowUsage >= 30) {
			id<MTLBuffer> smaller = copyTarget ? [device_ newBufferWithLength:copyTarget options:MTLResourceStorageModePrivate] : nil;
			if (!copyTarget || smaller) {
				copyPlane.buffer = smaller;
				if (smaller) {
					RecordTransientAllocation(TransientKind::COPY_PLANE, smaller.length);
				}
			}
			copyPlane.lowUsage = 0;
		}
	} else {
		copyPlane.lowUsage = 0;
	}
	copyPlane.used = 0;
	// Reuse command storage and upload memory only after GPU completion.
	[allocators_[nextSubmission_] reset];
	[initializationAllocators_[nextSubmission_] reset];
	[profileBuffers_[nextSubmission_] invalidateCounterRange:NSMakeRange(0, MAX_PROFILE_SAMPLES)];
	auto &arguments = argumentPools_[nextSubmission_];
	const size_t argumentTarget = std::max<size_t>(16, argumentUsage_[nextSubmission_] * 2);
	if (arguments.size() > argumentTarget) {
		if (++argumentLowUsage_[nextSubmission_] >= 30) {
			arguments.resize(argumentTarget);
			argumentLowUsage_[nextSubmission_] = 0;
		}
	} else {
		argumentLowUsage_[nextSubmission_] = 0;
	}
	argumentPoolUsed_ = 0;
	argumentUsage_[nextSubmission_] = 0;
	auto &blocks = uploadBlocks_[nextSubmission_];
	while (!blocks.empty() && blocks.back().used == 0) {
		blocks.pop_back();
	}
	for (auto &block : blocks) {
		// A slot has finished on the GPU. Retire a sustained allocation peak,
		// keeping headroom for ordinary fluctuations instead of resizing each frame.
		const size_t target = std::max<size_t>(1024 * 1024, block.used * 2);
		if (block.buffer.length > target * 2) {
			if (++block.lowUsage >= 30) {
				id<MTLBuffer> smaller = [device_ newBufferWithLength:target options:MTLResourceStorageModeShared];
				if (smaller) {
					block.buffer = smaller;
					++metrics_.uploadAllocations;
				}
				block.lowUsage = 0;
			}
		} else {
			block.lowUsage = 0;
		}
		block.used = 0;
	}
	currentUploadBlock_ = 0;
	viewMask_ = {};
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

id<MTLBuffer> RenderContext::AcquireCopyPlane(size_t size, std::string *error) {
	error->clear();
	if (!commands_ || !size || size > device_.maxBufferLength) {
		*error = "Invalid Metal copy plane size or command buffer";
		return nil;
	}
	auto &plane = copyPlanes_[nextSubmission_];
	if (!plane.buffer || plane.buffer.length < size) {
		id<MTLBuffer> buffer = [device_ newBufferWithLength:std::max<size_t>(4096, size) options:MTLResourceStorageModePrivate];
		if (!buffer) {
			*error = "Failed to allocate Metal depth/stencil copy buffer";
			return nil;
		}
		buffer.label = @"PPSSPP depth/stencil copy plane";
		plane.buffer = buffer;
		plane.lowUsage = 0;
		RecordTransientAllocation(TransientKind::COPY_PLANE, buffer.length);
	}
	plane.used = std::max(plane.used, size);
	++metrics_.copyPlaneRequests;
	Keep(plane.buffer);
	return plane.buffer;
}

ResourceMetrics RenderContext::Metrics() const {
	ResourceMetrics result = metrics_;
	for (const auto &blocks : uploadBlocks_) {
		for (const auto &block : blocks) {
			result.uploadCapacity += block.buffer.length;
		}
	}
	for (const auto &pool : argumentPools_) {
		result.argumentCapacity += pool.size();
	}
	for (const auto &plane : copyPlanes_) {
		result.copyPlaneCapacity += plane.buffer.length;
	}
	return result;
}

UploadSlice RenderContext::ReserveUpload(size_t size, std::string *error, size_t alignment) {
	error->clear();
	if (!device_ || !size || size > device_.maxBufferLength || !alignment ||
		alignment > 256 || (alignment & (alignment - 1)) != 0) {
		*error = "Invalid Metal transient upload size or device";
		return {};
	}
	if (!commands_) {
		*error = "Metal transient uploads require an active command buffer";
		return {};
	}
	++metrics_.uploadRequests;
	metrics_.uploadBytes += size;
	auto &blocks = uploadBlocks_[nextSubmission_];
	while (currentUploadBlock_ < blocks.size()) {
		auto &block = blocks[currentUploadBlock_];
		const size_t offset = (block.used + alignment - 1) & ~(alignment - 1);
		if (offset <= block.buffer.length && size <= block.buffer.length - offset) {
			metrics_.uploadPadding += offset - block.used;
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
	++metrics_.uploadAllocations;
	Keep(buffer);
	return {buffer, 0};
}

UploadSlice RenderContext::Upload(const void *data, size_t size, std::string *error, size_t alignment) {
	if (!data) {
		*error = "Invalid Metal transient upload data";
		return {};
	}
	auto slice = ReserveUpload(size, error, alignment);
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
	if (initializationCommands_ && !conservativeBarriers_) {
		auto boundary = BlitEncoder(initializationCommands_, "Initialization visibility", true);
		if (!boundary) {
			*error = "Failed to encode Metal initialization visibility";
			return false;
		}
		[boundary barrierAfterStages:MTLStageAll beforeQueueStages:MTLStageAll visibilityOptions:MTL4VisibilityOptionDevice];
		++metrics_.initializationBoundaries;
		EndEncoder(boundary);
	}
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
	submittedAccesses_[slot] = std::move(initializationAccesses_);
	MergeAccesses(&submittedAccesses_[slot], mainAccesses_);
	initializationAccesses_.clear();
	mainAccesses_.clear();
	submittedUnknownAccesses_[slot] = initializationUnknownAccesses_ || mainUnknownAccesses_;
	initializationUnknownAccesses_ = mainUnknownAccesses_ = false;
	// Initialization is submitted first, even when recorded after a main pass.
	std::stable_partition(profilePasses_.begin(), profilePasses_.end(), [](const ProfilePass &pass) { return pass.initialization; });
	submittedProfilePasses_[slot] = std::move(profilePasses_);
	submittedProfileSamplesUsed_[slot] = profileSamplesUsed_;
	submittedProfileLog_[slot] = profileLogEnabled_;
	submittedProfileIncomplete_[slot] = profileIncomplete_;
	profileIncomplete_ = false;
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
		const double libraryStart = time_now_d();
		id<MTLLibrary> library = [compiler_ newLibraryWithDescriptor:desc error:&compileError];
		metrics_.librarySeconds += time_now_d() - libraryStart;
		++metrics_.libraryCompiles;
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
