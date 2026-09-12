// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#import <Metal/Metal.h>

#include <array>
#include <string>
#include <vector>

#include "Common/GPU/Metal/MetalShaderCompiler.h"

namespace Metal {

struct UploadSlice {
	id<MTLBuffer> buffer = nil;
	size_t offset = 0;
	explicit operator bool() const { return buffer != nil; }
};

// Owned and encoded on the rendering thread. Command buffers retain the native
// resources they reference; CPU-visible upload memory must still be immutable
// until completion. Metal callbacks never touch the emulated GPU state.
class RenderContext {
public:
	RenderContext() = default;
	~RenderContext();
	RenderContext(const RenderContext &) = delete;
	RenderContext &operator=(const RenderContext &) = delete;

	bool Init(std::string *error);
	bool BeginCommands(std::string *error);
	// All encoders must have ended before submission. BLOCK readbacks submit with
	// wait=true and start a fresh command buffer before resuming rendering.
	bool SubmitCommands(bool wait, std::string *error);
	bool WaitUntilIdle(std::string *error);
	// Only initialize newly allocated resources here. These commands run before
	// the active render buffer, allowing uploads without interrupting its pass.
	// Updates to resources already referenced by a draw must use Commands().
	id<MTLCommandBuffer> InitializationCommands(std::string *error);
	// Transient data for the current command buffer. Encode its consumers before
	// submitting; cached slices are valid only for the same CommandGeneration().
	UploadSlice Upload(const void *data, size_t size, std::string *error);

	id<MTLDevice> Device() const { return device_; }
	id<MTLCommandBuffer> Commands() const { return commands_; }
	uint64_t CommandGeneration() const { return commandGeneration_; }
	std::string DeviceName() const;
	id<MTLFunction> CreateShader(const CompiledShader &shader, const char *tag, std::string *error);

private:
	static bool WaitForCommands(id<MTLCommandBuffer> commands, std::string *error);
	bool WaitForSubmission(size_t slot, std::string *error);

	id<MTLDevice> device_ = nil;
	id<MTLCommandQueue> queue_ = nil;
	id<MTLCommandBuffer> commands_ = nil;
	id<MTLCommandBuffer> initializationCommands_ = nil;
	std::array<id<MTLCommandBuffer>, 3> submitted_{};
	std::array<id<MTLCommandBuffer>, 3> submittedInitializations_{};
	struct UploadBlock {
		id<MTLBuffer> buffer = nil;
		size_t used = 0;
	};
	std::array<std::vector<UploadBlock>, 3> uploadBlocks_;
	size_t currentUploadBlock_ = 0;
	size_t nextSubmission_ = 0;
	uint64_t commandGeneration_ = 0;
};

}  // namespace Metal
