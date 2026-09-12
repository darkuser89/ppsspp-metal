// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Common/GPU/Metal/MetalRenderContext.h"

#include <algorithm>
#include <cstring>

namespace Metal {

RenderContext::~RenderContext() {
	// An unsubmitted buffer has no GPU work to drain. The owner must end its
	// encoders before destroying the context, even on initialization failure.
	commands_ = nil;
	initializationCommands_ = nil;
	std::string error;
	WaitUntilIdle(&error);
}

bool RenderContext::Init(std::string *error) {
	error->clear();
	if (device_) {
		*error = "Metal context is already initialized";
		return false;
	}
	if (@available(macOS 13.0, iOS 16.0, *)) {
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		if (!device || ![device supportsFamily:MTLGPUFamilyMetal3]) {
			*error = "This device does not support Metal 3";
			return false;
		}
		id<MTLCommandQueue> queue = [device newCommandQueue];
		if (!queue) {
			*error = "Failed to create the Metal command queue";
			return false;
		}
		queue.label = @"PPSSPP";
		device_ = device;
		queue_ = queue;
		return true;
	}
	*error = "Metal 3 requires macOS 13 or iOS 16 or later";
	return false;
}

std::string RenderContext::DeviceName() const {
	return device_ ? device_.name.UTF8String : "";
}

bool RenderContext::WaitForCommands(id<MTLCommandBuffer> commands, std::string *error) {
	if (!commands) {
		return true;
	}
	[commands waitUntilCompleted];
	if (commands.status == MTLCommandBufferStatusError) {
		const char *description = commands.error.localizedDescription.UTF8String;
		*error = description ? description : "Metal command buffer execution failed";
		return false;
	}
	return true;
}

bool RenderContext::WaitForSubmission(size_t slot, std::string *error) {
	bool success = WaitForCommands(submitted_[slot], error);
	std::string initializationError;
	if (!WaitForCommands(submittedInitializations_[slot], &initializationError)) {
		if (success) {
			*error = initializationError;
		}
		success = false;
	}
	submitted_[slot] = nil;
	submittedInitializations_[slot] = nil;
	return success;
}

id<MTLCommandBuffer> RenderContext::InitializationCommands(std::string *error) {
	error->clear();
	if (!commands_) {
		*error = "Metal resource initialization requires an active command buffer";
		return nil;
	}
	if (!initializationCommands_) {
		initializationCommands_ = [queue_ commandBuffer];
		if (!initializationCommands_) {
			*error = "Failed to allocate Metal resource initialization commands";
			return nil;
		}
		initializationCommands_.label = @"PPSSPP resource initialization";
	}
	return initializationCommands_;
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
	// This slot's GPU work has finished. Only now may its upload bytes be reused.
	auto &blocks = uploadBlocks_[nextSubmission_];
	while (!blocks.empty() && blocks.back().used == 0) {
		blocks.pop_back();
	}
	for (auto &block : blocks) {
		block.used = 0;
	}
	currentUploadBlock_ = 0;
	commands_ = [queue_ commandBuffer];
	if (!commands_) {
		*error = "Failed to allocate a Metal command buffer";
		return false;
	}
	commands_.label = @"PPSSPP render commands";
	++commandGeneration_;
	return true;
}

UploadSlice RenderContext::Upload(const void *data, size_t size, std::string *error) {
	error->clear();
	if (!device_ || !data || !size || size > device_.maxBufferLength) {
		*error = "Invalid Metal transient upload size, data or device";
		return {};
	}
	if (!commands_) {
		*error = "Metal transient uploads require an active command buffer";
		return {};
	}
	auto &blocks = uploadBlocks_[nextSubmission_];
	while (currentUploadBlock_ < blocks.size()) {
		auto &block = blocks[currentUploadBlock_];
		// One alignment works for both vertex and index bindings on Metal 3.
		const size_t offset = (block.used + 255) & ~size_t(255);
		if (offset <= block.buffer.length && size <= block.buffer.length - offset) {
			memcpy((uint8_t *)block.buffer.contents + offset, data, size);
			block.used = offset + size;
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
	memcpy(buffer.contents, data, size);
	blocks.push_back({buffer, size});
	return {buffer, 0};
}

bool RenderContext::SubmitCommands(bool wait, std::string *error) {
	error->clear();
	if (!commands_) {
		*error = "No active Metal command buffer to submit";
		return false;
	}
	id<MTLCommandBuffer> submitted = commands_;
	commands_ = nil;
	// The same queue preserves commit order. Initializations target only new
	// allocations, so they can precede every draw in this render buffer.
	if (initializationCommands_) {
		[initializationCommands_ commit];
	}
	[submitted commit];
	const size_t slot = nextSubmission_;
	submitted_[slot] = submitted;
	submittedInitializations_[slot] = initializationCommands_;
	initializationCommands_ = nil;
	nextSubmission_ = (nextSubmission_ + 1) % submitted_.size();
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

id<MTLFunction> RenderContext::CreateShader(const CompiledShader &shader, const char *tag, std::string *error) {
	error->clear();
	if (!device_ || shader.source.empty() || shader.entryPoint.empty()) {
		*error = "Metal shader or device is unavailable";
		return nil;
	}
	if (@available(macOS 13.0, iOS 16.0, *)) {
		@autoreleasepool {
			MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
			options.languageVersion = MTLLanguageVersion3_0;
			// PSP shaders use NaNs for range culling. Fast math may assume NaNs do
			// not occur, invalidating the shared generator's clipping behavior.
			options.fastMathEnabled = NO;
#if (defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000) || (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000)
			if (@available(macOS 15.0, iOS 18.0, *)) {
				// Permit the arithmetic optimizations used by the other hardware
				// backends without dropping the NaN/Inf semantics needed for culling.
				options.mathMode = MTLMathModeRelaxed;
			}
#endif
			// Honor the shared vertex generator's invariant position across shader
			// variants, including coplanar draws that depend on equal depth values.
			options.preserveInvariance = YES;
			NSString *source = [[NSString alloc] initWithBytes:shader.source.data() length:shader.source.size() encoding:NSUTF8StringEncoding];
			NSString *entryPoint = [[NSString alloc] initWithBytes:shader.entryPoint.data() length:shader.entryPoint.size() encoding:NSUTF8StringEncoding];
			if (!source || !entryPoint) {
				*error = "Metal shader source or entry point is not valid UTF-8";
				return nil;
			}
			NSError *compileError = nil;
			id<MTLLibrary> library = [device_ newLibraryWithSource:source options:options error:&compileError];
			if (!library) {
				const char *description = compileError.localizedDescription.UTF8String;
				*error = description ? description : "Metal library compilation failed";
				return nil;
			}
			if (tag) {
				library.label = [NSString stringWithUTF8String:tag];
			}
			id<MTLFunction> function = [library newFunctionWithName:entryPoint];
			if (!function) {
				*error = "Metal shader entry point was not found: " + shader.entryPoint;
			}
			return function;
		}
	}
	*error = "Metal Shading Language 3.0 is unavailable";
	return nil;
}

}  // namespace Metal
