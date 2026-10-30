// Copyright (c) 2012- PPSSPP Project.
// This program is free software, licensed under GPL-2.0-or-later.

// Standalone macOS regression tests against the production Metal objects.
#include "Common/GPU/metal4/thin3d_metal.h"
#include "Common/GPU/ShaderTranslation.h"
#include "Common/File/VFS/DirectoryReader.h"
#include "Common/File/VFS/VFS.h"
#include "Common/TimeUtil.h"
#include "Common/Log/LogManager.h"
#include "Core/Config.h"
#include "Core/ConfigValues.h"
#include "Core/CoreParameter.h"
#include "Core/System.h"
#include "Core/MemMap.h"
#include "Core/Util/PathUtil.h"
#include "GPU/metal4/DrawEngineMetal.h"
#include "GPU/metal4/GPUCommonHWMetal.h"
#include "GPU/metal4/GPU_Metal.h"
#include "GPU/metal4/PostShaderMetal.h"
#include "GPU/metal4/PresentationMetal.h"
#include "GPU/Common/Overlay.h"
#include "GPU/Common/SplineCommon.h"
#include "GPU/Common/SoftwareTransformCommon.h"
#include "GPU/Common/ShaderUniforms.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <mach/mach.h>
#include <thread>
#include "Common/File/FileUtil.h"
#include "Common/Data/Text/StringWriter.h"
#include "Core/ELF/ParamSFO.h"
#include "Common/Data/Format/PNGLoad.h"
#include "Common/Thread/ThreadManager.h"
#include <stdexcept>

@interface RegressionMetalLayer : CAMetalLayer
@property(nonatomic) NSUInteger attempts;
@property(nonatomic) BOOL unavailable;
@end
@implementation RegressionMetalLayer
- (id<CAMetalDrawable>)nextDrawable {
	++self.attempts;
	return self.unavailable ? nil : [super nextDrawable];
}
@end

static void Check(bool ok, const std::string &message) {
	if (!ok) {
		throw std::runtime_error(message);
	}
}

static void MultipleOutputs() {
	using namespace Draw;
	std::string error;
	std::unique_ptr<DrawContext> draw(T3DCreateMetalContext(&error));
	Check(draw && draw->GetDeviceCaps().multipleSwapchains, "native Metal multiple-output capability");
	auto *manager = (Metal::RenderManager *)draw->GetNativeObject(NativeObject::RENDER_MANAGER);
	draw->SetTargetSize(64, 40);
	int w, h;
	draw->GetFramebufferDimensions(draw->GetPrimarySwapchain(), &w, &h);
	Check(w == 64 && h == 40 && draw->GetPrimarySwapchain()->Width() == 64, "primary output dimensions");
	SwapchainDesc desc{WINDOWSYSTEM_METAL_EXT, nullptr, nullptr, 32, 24, "secondary regression"};
	auto *a = draw->CreateSwapchain(desc, &error);
	desc.width = 20;
	desc.height = 36;
	auto *b = draw->CreateSwapchain(desc, &error);
	Check(a && b && a != b, error);
	auto clear = [&](Framebuffer *fb, uint32_t color) {
		draw->BindFramebufferAsRenderTarget(fb, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR, color, 0.25f, 17}, "output regression");
	};
	auto verify = [&](Framebuffer *fb, uint32_t color) {
		draw->GetFramebufferDimensions(fb, &w, &h);
		std::vector<uint32_t> pixels(w * h);
		Check(draw->CopyFramebufferToMemory(fb, Aspect::COLOR_BIT, 0, 0, w, h, DataFormat::R8G8B8A8_UNORM,
			pixels.data(), w, ReadbackMode::BLOCK, "output pixels"), "output readback");
		Check(std::all_of(pixels.begin(), pixels.end(), [&](uint32_t pixel) { return pixel == color; }), "outputs shared or lost their contents");
	};
	for (int frame = 0; frame < 8; ++frame) {
		draw->BeginFrame(DebugFlags::NONE);
		clear(draw->GetPrimarySwapchain(), 0xFF102030 + frame);
		clear(a, 0xFF405060 + frame);
		clear(b, 0xFF708090 + frame);
		verify(a, 0xFF405060 + frame);
		verify(nullptr, 0xFF102030 + frame);
		verify(b, 0xFF708090 + frame);
		draw->Present(PresentMode::FIFO);
	}
	draw->ResizeSwapchain(a, 48, 16);
	draw->BeginFrame(DebugFlags::NONE);
	clear(a, 0xFFABCDEF);
	clear(b, 0xFF123456);
	verify(a, 0xFFABCDEF);
	verify(b, 0xFF123456);
	draw->Present(PresentMode::FIFO);
	draw->GetFramebufferDimensions(a, &w, &h);
	Check(w == 48 && h == 16, "secondary resize used the primary size");
	draw->DestroySwapchain(a);
	// The secondary keeps rendering without acquiring or binding the primary.
	draw->SetTargetSize(0, 0);
	draw->BeginFrame(DebugFlags::NONE);
	clear(b, 0xFFEECCAA);
	verify(b, 0xFFEECCAA);
	draw->Present(PresentMode::FIFO);
	draw->DestroySwapchain(b);
	desc.width = 0;
	Check(!draw->CreateSwapchain(desc, &error) && !error.empty(), "zero-size output accepted");
	desc.width = draw->GetDeviceCaps().maxTextureSize + 1;
	Check(!draw->CreateSwapchain(desc, &error), "oversized output accepted");

	RegressionMetalLayer *hidden = [RegressionMetalLayer layer];
	hidden.unavailable = YES;
	hidden.drawableSize = CGSizeMake(16, 16);
	desc = {WINDOWSYSTEM_METAL_EXT, (__bridge void *)hidden, nullptr, 0, 0, "unavailable regression"};
	a = draw->CreateSwapchain(desc, &error);
	Check(a != nullptr, error);
	draw->BeginFrame(DebugFlags::NONE);
	clear(a, 0xFF123456);
	clear(a, 0xFF345678);
	draw->ResizeSwapchain(a, 24, 24);
	clear(a, 0xFF567890);
	Check(hidden.attempts == 1, "unavailable drawable requested twice in one frame");
	draw->Present(PresentMode::FIFO);
	draw->BeginFrame(DebugFlags::NONE);
	clear(a, 0xFF123456);
	Check(hidden.attempts == 2, "unavailable output was not retried next frame");
	draw->Present(PresentMode::FIFO);
	draw->DestroySwapchain(a);
	Check(manager->Context().WaitUntilIdle(&error), error);
	std::cout << "PASS multiple outputs: eight three-target frames, exact GPU pixels, resize/destruction, primary absent, unavailable acquisition once per frame\n";

	RegressionMetalLayer *layerA = [RegressionMetalLayer layer];
	RegressionMetalLayer *layerB = [RegressionMetalLayer layer];
	layerA.drawableSize = CGSizeMake(16, 16);
	layerB.drawableSize = CGSizeMake(24, 16);
	desc = {WINDOWSYSTEM_METAL_EXT, (__bridge void *)layerA, nullptr, 16, 16, "real output A"};
	a = draw->CreateSwapchain(desc, &error);
	desc.data1 = (__bridge void *)layerB;
	desc.width = 24;
	b = draw->CreateSwapchain(desc, &error);
	Check(a && b, error);
	const auto presents = manager->Context().Metrics().presentedDrawables;
	for (int frame = 0; frame < 2; ++frame) {
		if (frame == 1) {
			// A surface can change size without the platform having sent its resize event yet.
			layerA.drawableSize = CGSizeMake(32, 24);
			layerB.drawableSize = CGSizeMake(8, 32);
		}
		draw->BeginFrame(DebugFlags::NONE);
		clear(a, 0xFF0055AA);
		verify(a, 0xFF0055AA);
		clear(b, 0xFFAA5500);
		verify(b, 0xFFAA5500);
		Check(a->Width() == (frame ? 32 : 16) && b->Height() == (frame ? 32 : 16), "actual drawable size did not replace the stale output extent");
		draw->Present(PresentMode::FIFO);
		Check(manager->Context().WaitUntilIdle(&error), error);
	}
	Check(layerA.attempts == 2 && layerB.attempts == 2 && manager->Context().Metrics().presentedDrawables == presents + 4,
		"all acquired native drawables must be presented once after readback submission");
	draw->DestroySwapchain(a);
	draw->DestroySwapchain(b);
	std::cout << "PASS native drawable outputs: two CAMetalLayers, exact GPU pixels, unnotified size change, readback submissions then four queue presentations\n";
}

static void DependencyAccounting() {
	const uint64_t stages[] = {MTLStageVertex, MTLStageFragment, MTLStageBlit, MTLStageDispatch};
	unsigned cases = 0;
	for (uint64_t producer : stages) {
		for (uint64_t consumer : stages) {
			for (unsigned mode = 0; mode < 4; ++mode) {
				const bool priorWrite = mode & 1;
				const bool nextWrite = mode & 2;
				Metal::ResourceAccesses prior{{1, {priorWrite ? 0 : producer, priorWrite ? producer : 0}}};
				Metal::ResourceAccesses current{{1, {nextWrite ? 0 : consumer, nextWrite ? consumer : 0}}};
				const auto hazards = Metal::FindDependencies(prior, current);
				Check(hazards.raw == (priorWrite && !nextWrite) && hazards.war == (!priorWrite && nextWrite) &&
					hazards.waw == (priorWrite && nextWrite), "dependency classification");
				Check(hazards.afterStages == (mode ? producer : 0) && hazards.beforeStages == (mode ? consumer : 0), "dependency stages");
				current[2] = current[1];
				current.erase(1);
				Check(Metal::FindDependencies(prior, current).afterStages == 0, "independent allocation has no dependency");
				++cases;
			}
		}
	}
	Metal::ResourceAccesses history{{1, {MTLStageVertex, 0}}};
	Metal::MergeAccesses(&history, {{1, {MTLStageFragment, 0}}});
	auto overwrite = Metal::FindDependencies(history, {{1, {0, MTLStageBlit}}});
	Check(overwrite.afterStages == (MTLStageVertex | MTLStageFragment) && overwrite.war == 1, "all reader stages retained");
	std::cout << "PASS dependency accounting " << cases << " stage/access pairs, independent allocations and multiple readers\n";

	Metal::RenderContext context;
	std::string error;
	Check(context.Init(&error, 2), error);
	context.SetProfilingEnabled(true, false);
	Check(context.BeginCommands(&error), error);
	const uint32_t pattern = 0x1234abcd;
	auto upload = context.Upload(&pattern, sizeof(pattern), &error, 4);
	auto a = [context.Device() newBufferWithLength:4 options:MTLResourceStorageModePrivate];
	auto b = [context.Device() newBufferWithLength:4 options:MTLResourceStorageModePrivate];
	auto output = [context.Device() newBufferWithLength:4 options:MTLResourceStorageModeShared];
	Check(upload && a && b && output, "dependency fixture allocations");
	context.Keep({a, b, output});
	auto copy = [&](id<MTL4CommandBuffer> commands, id<MTLBuffer> src, size_t offset, id<MTLBuffer> dst) {
		auto encoder = context.BlitEncoder(commands, "Dependency fixture", true);
		Check(encoder != nil, "dependency fixture encoder");
		context.UseResource(encoder, src, MTLStageBlit, (MTLStages)0);
		context.UseResource(encoder, dst, (MTLStages)0, MTLStageBlit);
		[encoder copyFromBuffer:src sourceOffset:offset toBuffer:dst destinationOffset:0 size:4];
		context.EndEncoder(encoder);
	};
	copy(context.Commands(), a, 0, b);
	// Recorded second, executed first. GPU output and dependency trace must agree.
	copy(context.InitializationCommands(&error), upload.buffer, upload.offset, a);
	copy(context.Commands(), b, 0, a);
	copy(context.Commands(), a, 0, output);
	Check(context.SubmitCommands(true, &error), error);
	Check(*(const uint32_t *)output.contents == pattern, "initialization/main GPU order");
	const auto &profile = context.LastDependencyProfile();
	Check(profile.passes == 5 && profile.initializationPasses == 2 && profile.resourceAccesses == 8, "dependency pass coverage");
	Check(profile.hazards.raw == 4 && profile.hazards.war == 1 && profile.hazards.waw == 1 && profile.initializationToMainHazards == 3,
		"initialization/main dependency counts");
	Check(!profile.incomplete, "dependency fixture profile complete");
	Check(context.Metrics().completionWaits == 1 && context.Metrics().completionWaitSeconds >= 0.0, "completion wait accounting");
	std::cout << "PASS dependency GPU order: later-recorded initialization, RAW=4 WAR=1 WAW=1, initialization-to-main=3\n";

	Check(context.BeginCommands(&error), error);
	auto backing = [context.Device() newBufferWithLength:256 options:MTLResourceStorageModeShared];
	auto aliasedOutput = [context.Device() newBufferWithLength:512 options:MTLResourceStorageModeShared];
	Check(backing && aliasedOutput, "alias buffers");
	*(uint32_t *)backing.contents = pattern;
	auto descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:1 height:1 mipmapped:NO];
	descriptor.storageMode = MTLStorageModeShared;
	descriptor.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	auto texture = [backing newTextureWithDescriptor:descriptor offset:0 bytesPerRow:256];
	auto view = [texture newTextureViewWithPixelFormat:MTLPixelFormatRGBA8Unorm];
	Check(texture && view, "buffer-backed texture view");
	context.Keep({backing, texture, view, aliasedOutput});
	auto encoder = context.BlitEncoder(context.Commands(), "Alias dependency fixture", true);
	context.UseResource(encoder, backing, MTLStageBlit, (MTLStages)0);
	context.UseResource(encoder, view, MTLStageBlit, (MTLStages)0);
	context.UseResource(encoder, aliasedOutput, (MTLStages)0, MTLStageBlit);
	const auto identities = context.Metrics().resourceIdentityResolutions;
	for (int i = 0; i < 10000; ++i) {
		context.UseResource(encoder, backing, MTLStageBlit, (MTLStages)0);
		context.UseResource(encoder, view, MTLStageBlit, (MTLStages)0);
	}
	Check(context.Metrics().resourceIdentityResolutions == identities, "repeated binds resolved resource identity again");
	[encoder copyFromBuffer:backing sourceOffset:0 toBuffer:aliasedOutput destinationOffset:0 size:4];
	[encoder copyFromTexture:view sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
		sourceSize:MTLSizeMake(1, 1, 1) toBuffer:aliasedOutput destinationOffset:256 destinationBytesPerRow:256 destinationBytesPerImage:256];
	context.EndEncoder(encoder);
	Check(context.SubmitCommands(true, &error), error);
	Check(*(const uint32_t *)aliasedOutput.contents == pattern && *(const uint32_t *)((const uint8_t *)aliasedOutput.contents + 256) == pattern,
		"buffer/view GPU reads");
	Check(context.LastDependencyProfile().passes == 1 && context.LastDependencyProfile().resourceAccesses == 2,
		"texture views and backing buffer share one dependency identity");
	std::cout << "PASS dependency alias identity: buffer and texture view, two exact GPU readbacks\n";
	Check(context.BeginCommands(&error), error);
	encoder = context.BlitEncoder(context.Commands(), "New submission identity", true);
	context.UseResource(encoder, backing, MTLStageBlit, (MTLStages)0);
	Check(context.Metrics().resourceIdentityResolutions == identities + 1, "resource identity survived submission recycling");
	context.EndEncoder(encoder);
	Check(context.SubmitCommands(true, &error), error);
	std::cout << "PASS resource identity cache: 20000 repeated binds, alias equivalence, submission invalidation\n";
}

static void OutputRetirement(bool destroy) {
	using namespace Draw;
	for (int variant = 0; variant < 4; ++variant) {
		const bool native = variant & 1;
		const bool afterReadback = variant & 2;
		std::string error;
		std::unique_ptr<DrawContext> draw(T3DCreateMetalContext(&error));
		Check(draw != nullptr, error);
		auto *manager = (Metal::RenderManager *)draw->GetNativeObject(NativeObject::RENDER_MANAGER);
		auto &context = manager->Context();
		RegressionMetalLayer *layer = native ? [RegressionMetalLayer layer] : nil;
		layer.drawableSize = CGSizeMake(16, 16);
		SwapchainDesc desc{WINDOWSYSTEM_METAL_EXT, (__bridge void *)layer, nullptr, 16, 16, "retiring output"};
		auto *output = draw->CreateSwapchain(desc, &error);
		Check(output != nullptr, error);
		draw->BeginFrame(DebugFlags::NONE);
		draw->BindFramebufferAsRenderTarget(output, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
			0xFF123456, 0.375f, 91}, "retiring output");
		manager->EndRenderPass();
		__weak id<MTLTexture> oldColor = static_cast<Metal::Framebuffer *>(manager->RenderTarget())->Color();
		__weak id<MTLTexture> oldDepth = static_cast<Metal::Framebuffer *>(manager->RenderTarget())->DepthStencil();
		if (afterReadback) {
			std::array<uint32_t, 16 * 16> pixels{};
			Check(draw->CopyFramebufferToMemory(output, Aspect::COLOR_BIT, 0, 0, 16, 16, DataFormat::R8G8B8A8_UNORM,
				pixels.data(), 16, ReadbackMode::BLOCK, "retiring pixels"), "retiring output readback");
			Check(pixels[0] == 0xFF123456 && !context.Commands(), "readback did not finish the submission");
		}
		auto commands = context.Commands();
		const auto waits = context.Metrics().completionWaits;
		const auto presents = context.Metrics().presentedDrawables;
		if (destroy) {
			draw->DestroySwapchain(output);
		} else {
			draw->ResizeSwapchain(output, 24, 32);
		}
		std::cout << "MEASURE output-retirement destroy=" << destroy << " native=" << native << " after-readback=" << afterReadback
			<< " completion-waits=" << context.Metrics().completionWaits - waits << '\n';
		Check(context.Commands() == commands && context.Metrics().completionWaits == waits,
			"output retirement submitted or waited for unrelated GPU work");
		Check(afterReadback || (oldColor && oldDepth), "retired attachments released before GPU submission");
		draw->Present(PresentMode::IMMEDIATE);
		Check(context.WaitUntilIdle(&error), error);
		Check(context.Metrics().presentedDrawables == presents + native, "retired native drawable lost its presentation");
		if (!destroy) {
			if (native) {
				layer.drawableSize = CGSizeMake(24, 32);
			}
			draw->BeginFrame(DebugFlags::NONE);
			draw->BindFramebufferAsRenderTarget(output, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
				0xFFABCDEF, 0.625f, 173}, "resized output");
			std::vector<uint32_t> pixels(24 * 32);
			Check(draw->CopyFramebufferToMemory(output, Aspect::COLOR_BIT, 0, 0, 24, 32, DataFormat::R8G8B8A8_UNORM,
				pixels.data(), 24, ReadbackMode::BLOCK, "resized pixels"), "resized output readback");
			Check(std::all_of(pixels.begin(), pixels.end(), [](uint32_t pixel) { return pixel == 0xFFABCDEF; }),
				"retired output contaminated its replacement");
			draw->Present(PresentMode::IMMEDIATE);
			draw->DestroySwapchain(output);
		}
		Check(context.WaitUntilIdle(&error), error);
	}
	std::cout << "PASS nonblocking output " << (destroy ? "destruction" : "resize")
		<< ": offscreen/native attachments retained, no submission/wait, drawable presented, replacement pixels\n";
}

static void DynamicDepthAttachments() {
	using namespace Draw;
	std::string error;
	std::unique_ptr<DrawContext> draw(T3DCreateMetalContext(&error));
	Check(draw != nullptr, error);
	const char *vertexSource = R"(#version 450
layout(location=0) in vec3 position;
layout(location=1) in vec4 color;
layout(location=0) out vec4 varyingColor;
void main() { gl_Position = vec4(position, 1.0); varyingColor = color; }
)";
	const char *fragmentSource = R"(#version 450
layout(location=0) in vec4 varyingColor;
layout(location=0) out vec4 color;
void main() { color = varyingColor; }
)";
	AutoRef<ShaderModule> vs(draw->CreateShaderModule(ShaderStage::Vertex, GLSL_VULKAN,
		(const uint8_t *)vertexSource, strlen(vertexSource), "dynamic depth vertex"));
	AutoRef<ShaderModule> fs(draw->CreateShaderModule(ShaderStage::Fragment, GLSL_VULKAN,
		(const uint8_t *)fragmentSource, strlen(fragmentSource), "dynamic depth fragment"));
	struct Vertex { float x, y, z, r, g, b, a; };
	AutoRef<InputLayout> layout(draw->CreateInputLayout({sizeof(Vertex), {
		{0, DataFormat::R32G32B32_FLOAT, 0}, {1, DataFormat::R32G32B32A32_FLOAT, 12}}}));
	DepthStencilStateDesc depthDesc{true, true, Comparison::LESS, true,
		{StencilOp::KEEP, StencilOp::REPLACE, StencilOp::KEEP, Comparison::ALWAYS}};
	AutoRef<DepthStencilState> depth(draw->CreateDepthStencilState(depthDesc));
	AutoRef<BlendState> blend(draw->CreateBlendState({false, 0xF}));
	AutoRef<RasterState> raster(draw->CreateRasterState({}));
	Check(vs && fs && layout && depth && blend && raster, "dynamic depth pipeline resources");
	PipelineDesc desc{Primitive::TRIANGLE_LIST, {vs, fs}, layout, depth, blend, raster, nullptr};
	AutoRef<Pipeline> pipeline(draw->CreateGraphicsPipeline(desc, "dynamic depth"));
	Check(pipeline != nullptr, "dynamic depth pipeline");
	int cases = 0;
	for (int samples : {0, 2}) {
		for (bool hasDepth : {true, false, true, false}) {
			AutoRef<Framebuffer> fb(draw->CreateFramebuffer({8, 8, 1, 1, samples, hasDepth, "dynamic depth target"}));
			Check(fb != nullptr, "dynamic depth framebuffer");
			draw->BeginFrame(DebugFlags::NONE);
			draw->BindFramebufferAsRenderTarget(fb, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
				0, 0.5f, 17}, "dynamic depth");
			draw->BindPipeline(pipeline);
			draw->SetViewport({0, 0, 8, 8, 0, 1});
			draw->SetScissorRect(0, 0, 8, 8);
			for (bool far : {false, true}) {
				const float z = far ? 0.75f : 0.25f;
				const float r = far ? 1.0f : 0.0f, g = far ? 0.0f : 1.0f;
				const Vertex vertices[] = {{-1, -1, z, r, g, 0, 1}, {3, -1, z, r, g, 0, 1}, {-1, 3, z, r, g, 0, 1}};
				draw->SetStencilParams(far ? 173 : 91, 255, 255);
				draw->DrawUP(vertices, 3);
			}
			std::array<uint32_t, 64> colors{};
			Check(draw->CopyFramebufferToMemory(fb, Aspect::COLOR_BIT, 0, 0, 8, 8,
				DataFormat::R8G8B8A8_UNORM, colors.data(), 8, ReadbackMode::BLOCK, "dynamic depth colors"), "dynamic depth colors");
			const uint32_t expected = hasDepth ? 0xFF00FF00 : 0xFF0000FF;
			Check(std::all_of(colors.begin(), colors.end(), [&](uint32_t color) { return color == expected; }),
				"depth attachment did not select dynamic depth/stencil state");
			if (hasDepth) {
				std::array<float, 64> depths{};
				std::array<uint8_t, 64> stencils{};
				Check(draw->CopyFramebufferToMemory(fb, Aspect::DEPTH_BIT, 0, 0, 8, 8,
					DataFormat::D32F, depths.data(), 8, ReadbackMode::BLOCK, "dynamic depth values"), "dynamic depth values");
				Check(draw->CopyFramebufferToMemory(fb, Aspect::STENCIL_BIT, 0, 0, 8, 8,
					DataFormat::S8, stencils.data(), 8, ReadbackMode::BLOCK, "dynamic stencil values"), "dynamic stencil values");
				Check(std::all_of(depths.begin(), depths.end(), [](float z) { return z == 0.25f; }) &&
					std::all_of(stencils.begin(), stencils.end(), [](uint8_t s) { return s == 91; }), "dynamic depth/stencil writes");
			}
			draw->Present(PresentMode::IMMEDIATE);
			++cases;
		}
	}
	std::cout << "PASS dynamic depth attachments: " << cases << " color/depth/stencil cases, one pipeline object, 1x/MSAA4\n";
}

static void PrimarySurfaceChanges(int change) {
	using namespace Draw;
	for (int variant = 0; variant < 2; ++variant) {
		std::string error;
		std::unique_ptr<DrawContext> draw(T3DCreateMetalContext(&error));
		Check(draw != nullptr, error);
		auto *manager = (Metal::RenderManager *)draw->GetNativeObject(NativeObject::RENDER_MANAGER);
		auto &context = manager->Context();
		RegressionMetalLayer *a = [RegressionMetalLayer layer], *b = [RegressionMetalLayer layer];
		a.drawableSize = b.drawableSize = CGSizeMake(16, 16);
		Check(manager->SetSurface(a, &error), error);
		draw->BeginFrame(DebugFlags::NONE);
		draw->BindFramebufferAsRenderTarget(nullptr, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
			0xFF123456, 0.25f, 91}, "primary change");
		manager->EndRenderPass();
		if (variant) {
			std::array<uint32_t, 256> pixels{};
			Check(draw->CopyFramebufferToMemory(nullptr, Aspect::COLOR_BIT, 0, 0, 16, 16,
				DataFormat::R8G8B8A8_UNORM, pixels.data(), 16, ReadbackMode::BLOCK, "primary before change"), "primary pixels");
			Check(pixels[0] == 0xFF123456, "old primary color");
		}
		const auto waits = context.Metrics().completionWaits;
		const auto presents = context.Metrics().presentedDrawables;
		auto commands = context.Commands();
		if (change == 0) {
			Check(manager->SetSurface(b, &error), error);
		} else if (change == 1) {
			Check(manager->SetSurface(nil, &error), error);
		} else if (change == 2) {
			a.drawableSize = CGSizeMake(24, 32);
			manager->ResizeSurface();
		} else if (change == 6) {
			Check(manager->SetSurface(a, &error), error);
		} else {
			a.drawableSize = CGSizeMake(24, 32);
			const Event event = change == 3 ? Event::RESIZED : change == 4 ? Event::LOST_BACKBUFFER : Event::GOT_BACKBUFFER;
			draw->HandleEvent(event, 24, 32, nullptr, nullptr);
		}
		std::cout << "MEASURE primary-change action=" << change << " after-readback=" << variant
			<< " completion-waits=" << context.Metrics().completionWaits - waits << '\n';
		Check(context.Commands() == commands && context.Metrics().completionWaits == waits,
			"primary change submitted or drained existing GPU work");
		draw->Present(PresentMode::IMMEDIATE);
		Check(context.WaitUntilIdle(&error), error);
		Check(context.Metrics().presentedDrawables == presents + 1, "retired primary drawable was dropped");
		draw->BeginFrame(DebugFlags::NONE);
		draw->BindFramebufferAsRenderTarget(nullptr, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
			0xFFABCDEF, 0.75f, 173}, "new primary");
		int w, h;
		draw->GetFramebufferDimensions(nullptr, &w, &h);
		std::vector<uint32_t> pixels(w * h);
		Check(draw->CopyFramebufferToMemory(nullptr, Aspect::COLOR_BIT, 0, 0, w, h,
			DataFormat::R8G8B8A8_UNORM, pixels.data(), w, ReadbackMode::BLOCK, "new primary"), "new primary readback");
		Check(std::all_of(pixels.begin(), pixels.end(), [](uint32_t p) { return p == 0xFFABCDEF; }), "new primary contents");
		draw->Present(PresentMode::IMMEDIATE);
		Check(context.WaitUntilIdle(&error), error);
	}
	std::cout << "PASS primary surface change " << change << ": active/after-readback, no forced wait, old drawable presented and new output exact\n";
}

static void PeriodicCacheSaves() {
	const auto oldConfig = g_Config;
	const auto oldCore = PSP_CoreParameter();
	const auto oldState = gstate;
	const auto oldCached = gstate_c;
	const auto oldStats = gpuStats;
	const auto oldSFO = g_paramSFO;
	auto *oldGPU = gpu;
	struct Restore {
		std::function<void()> apply;
		~Restore() { apply(); }
	} restore{[&] { g_Config = oldConfig; PSP_CoreParameter() = oldCore; gstate = oldState; gstate_c = oldCached;
		gpuStats = oldStats; g_paramSFO = oldSFO; gpu = oldGPU; }};
	g_Config.appCacheDirectory = Path(std::filesystem::current_path().string() + "/periodic-cache");
	g_Config.memStickDirectory = g_Config.appCacheDirectory;
	g_Config.bShaderCache = true;
	g_Config.bSkipBufferEffects = false;
	g_Config.bStereoRendering = false;
	g_Config.bMetalFXSpatial = false;
	g_Config.iInternalResolution = 1;
	g_Config.iMultiSampleLevel = 0;
	g_Config.vPostShaderNames.clear();
	PSP_CoreParameter().renderWidth = 480;
	PSP_CoreParameter().renderHeight = 272;
	PSP_CoreParameter().renderScaleFactor = 1;
	g_paramSFO.Clear();
	g_paramSFO.SetValue("DISC_ID", "M4TEST001", 10);
	std::filesystem::remove_all(g_Config.appCacheDirectory.ToString());
	std::string error;
	std::unique_ptr<Draw::DrawContext> draw(Draw::T3DCreateMetalContext(&error));
	Check(draw && draw->CreatePresets(), "periodic cache context");
	std::unique_ptr<GPUCommon> renderer(CreateMetalGPU(nullptr, draw.get()));
	gpu = renderer.get();
	DisplayLayoutConfig config{};
	const std::pair<int, unsigned long long> sequence[] = {{0, 0}, {32767, 0}, {32768, 1}, {32768, 1},
		{32768, 1}, {32769, 1}, {65536, 2}, {65536, 2}, {0, 2}, {32768, 3}, {32768, 3}};
	for (auto [flips, expected] : sequence) {
		gpuStats.totals.numFlips = flips;
		renderer->BeginHostFrame(config);
		char text[8192];
		StringWriter writer(text);
		renderer->GetStats(writer);
		const char *line = strstr(text, "Metal pipeline cache saves:");
		double ms = 0;
		unsigned long long count = UINT64_MAX;
		Check(line && sscanf(line, "Metal pipeline cache saves: %lf ms (%llu", &ms, &count) == 2, "periodic cache statistics");
		std::cout << "MEASURE periodic-cache flips=" << flips << " saves=" << count << '\n';
		Check(count == expected, "repeated host frames rewrote caches at the same PSP flip");
		if (expected) {
			Check(VerifyShaderCacheChecksum(g_Config.appCacheDirectory / "M4TEST001.metalpipelinecache"), "periodic pipeline cache not written completely");
		}
	}
	std::cout << "PASS periodic cache save: real GPU host-frame path, repeated flips, next boundary, counter reset and valid pipeline file\n";
}

static void DrawableReadbackOrdering() {
	using namespace Draw;
	for (bool primary : {true, false}) {
		std::string error;
		std::unique_ptr<DrawContext> draw(T3DCreateMetalContext(&error));
		Check(draw != nullptr, error);
		auto *manager = (Metal::RenderManager *)draw->GetNativeObject(NativeObject::RENDER_MANAGER);
		auto &context = manager->Context();
		RegressionMetalLayer *layer = [RegressionMetalLayer layer];
		layer.drawableSize = CGSizeMake(16, 16);
		Swapchain *secondary = nullptr;
		if (primary) {
			Check(manager->SetSurface(layer, &error), error);
		} else {
			secondary = draw->CreateSwapchain({WINDOWSYSTEM_METAL_EXT, (__bridge void *)layer, nullptr, 16, 16, "drawable ordering"}, &error);
			Check(secondary != nullptr, error);
		}
		draw->BeginFrame(DebugFlags::NONE);
		draw->BindFramebufferAsRenderTarget(secondary, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
			0xFF123456, 0.25f, 91}, "drawable before readback");
		for (int read = 0; read < 2; ++read) {
			std::array<uint32_t, 256> pixels{};
			Check(draw->CopyFramebufferToMemory(secondary, Aspect::COLOR_BIT, 0, 0, 16, 16,
				DataFormat::R8G8B8A8_UNORM, pixels.data(), 16, ReadbackMode::BLOCK, "drawable ordering readback"), "drawable readback");
			std::cout << "MEASURE drawable-order primary=" << primary << " read=" << read
				<< " queue-waits=" << context.Metrics().drawableWaits << " presents=" << context.Metrics().presentedDrawables << '\n';
			Check(context.Metrics().drawableWaits == 1 && context.Metrics().presentedDrawables == 0,
				"drawable access submitted before its queue wait, or presented before the final frame");
			Check(std::all_of(pixels.begin(), pixels.end(), [](uint32_t pixel) { return pixel == 0xFF123456; }), "ordered drawable pixels");
		}
		draw->Present(PresentMode::IMMEDIATE);
		Check(context.WaitUntilIdle(&error), error);
		Check(context.Metrics().drawableWaits == 1 && context.Metrics().presentedDrawables == 1,
			"final presentation repeated the acquisition wait or missed its drawable");
		if (secondary) {
			draw->DestroySwapchain(secondary);
		}
	}
	std::cout << "PASS drawable ordering: primary/secondary, wait before first readback, no repeat wait, exact pixels, one final presentation\n";
}

static void SubmissionDependencies() {
	Metal::RenderContext context;
	std::string error;
	Check(context.Init(&error, 3), error);
	auto a = [context.Device() newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
	auto b = [context.Device() newBufferWithLength:4096 options:MTLResourceStorageModePrivate];
	Check(a && b, "submission dependency buffers");
	std::vector<id<MTLBuffer>> outputs;
	unsigned pending = 0;
	for (uint32_t iteration = 0; iteration < 96; ++iteration) {
		Check(context.BeginCommands(&error), error);
		std::array<uint32_t, 1024> values;
		for (size_t i = 0; i < values.size(); ++i) {
			values[i] = iteration * 1000003 + (uint32_t)i;
		}
		auto upload = context.Upload(values.data(), sizeof(values), &error);
		auto output = [context.Device() newBufferWithLength:sizeof(values) options:MTLResourceStorageModeShared];
		Check(upload && output, "submission dependency upload/readback");
		outputs.push_back(output);
		auto copy = [&](id<MTL4CommandBuffer> commands, id<MTLBuffer> src, size_t offset, id<MTLBuffer> dst, bool declared) {
			auto encoder = context.BlitEncoder(commands, "Submission dependency fixture", declared);
			Check(encoder != nil, "submission dependency encoder");
			context.Keep({src, dst});
			if (declared) {
				context.UseResource(encoder, src, MTLStageBlit, (MTLStages)0);
				context.UseResource(encoder, dst, (MTLStages)0, MTLStageBlit);
			}
			[encoder copyFromBuffer:src sourceOffset:offset toBuffer:dst destinationOffset:0 size:sizeof(values)];
			context.EndEncoder(encoder);
		};
		// Alternate early initialization and ordinary main writes to the same
		// allocation across three in-flight slots; occasionally use opaque work.
		copy(iteration & 1 ? context.InitializationCommands(&error) : context.Commands(), upload.buffer, upload.offset, a, iteration % 11 != 0);
		copy(context.Commands(), a, 0, b, true);
		copy(context.Commands(), b, 0, output, true);
		auto completion = context.CurrentSubmission();
		Check(context.SubmitCommands(false, &error), error);
		pending += !completion->IsComplete();
	}
	Check(context.WaitUntilIdle(&error), error);
	for (uint32_t iteration = 0; iteration < outputs.size(); ++iteration) {
		const auto *values = (const uint32_t *)outputs[iteration].contents;
		for (uint32_t i = 0; i < 1024; ++i) {
			Check(values[i] == iteration * 1000003 + i, "asynchronous submission dependency contents");
		}
	}
	const auto metrics = context.Metrics();
	Check(pending > 0 && metrics.priorSubmissionDependencies > 0, "prior submissions actually tracked");
	Check(metrics.initializationBoundaries == 48 && metrics.fullQueueBarriers == 9, "init boundaries and opaque fallback");
	std::cout << "PASS submission dependencies: 96 async submissions, " << pending << " observed pending, "
		<< metrics.priorSubmissionDependencies << " cross-submission hazards, 48 init boundaries, 9 opaque fallbacks\n";
}

static void DependencyTimings(bool profile) {
	Metal::RenderContext context;
	std::string error;
	Check(context.Init(&error, 2), error);
	const char *measurement = profile ? "dependency-pass-timing" : "dependency";
	Check(context.TimestampFrequency() > 0, "GPU timestamp frequency");
	std::cout << "MEASURE " << measurement << " timestamp-frequency=" << context.TimestampFrequency() << " ticks-per-second\n";
	Metal::CompiledShader shader;
	shader.entryPoint = "dependency_bench";
	shader.source = R"(#include <metal_stdlib>
using namespace metal;
kernel void dependency_bench(constant uint &seed [[buffer(0)]], device uint *output [[buffer(1)]], uint id [[thread_position_in_grid]]) {
	uint x = seed + id;
	for (uint i = 0; i < 128; ++i) {
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
	}
	output[id] = x;
})";
	auto function = context.CreateShader(shader, "Dependency timing fixture", &error);
	Check(function != nil, error);
	NSError *nativeError = nil;
	auto pipeline = context.CreateComputePipeline(function, &nativeError);
	Check(pipeline != nil, nativeError ? nativeError.localizedDescription.UTF8String : "dependency compute pipeline");
	MTL4CounterHeapDescriptor *counterDesc = [MTL4CounterHeapDescriptor new];
	counterDesc.type = MTL4CounterHeapTypeTimestamp;
	counterDesc.count = 34;
	id<MTL4CounterHeap> counters = profile ? [context.Device() newCounterHeapWithDescriptor:counterDesc error:&nativeError] : nil;
	Check(!profile || counters != nil, "dependency timing counter heap");
	constexpr size_t VERTICES = 32768;
	constexpr size_t COPY_BYTES = 8 * 1024 * 1024;
	auto output = [context.Device() newBufferWithLength:VERTICES * sizeof(uint32_t) options:MTLResourceStorageModeShared];
	auto source = [context.Device() newBufferWithLength:COPY_BYTES options:MTLResourceStorageModeShared];
	auto destination = [context.Device() newBufferWithLength:COPY_BYTES options:MTLResourceStorageModePrivate];
	auto readback = [context.Device() newBufferWithLength:COPY_BYTES options:MTLResourceStorageModeShared];
	Check(output && source && destination && readback, "dependency benchmark buffers");
	struct Sample { double gpu, encode, wait, computePasses, blitPasses; };
	std::vector<Sample> samples[2];
	uint64_t barriers[2]{}, full[2]{};
	std::vector<uint32_t> expected(VERTICES);
	for (uint32_t round = 0; round < 80; ++round) {
		const uint32_t seed = 0x31415926 + round * 19;
		for (uint32_t id = 0; id < VERTICES; ++id) {
			uint32_t x = seed + id;
			for (unsigned i = 0; i < 128; ++i) {
				x ^= x << 13;
				x ^= x >> 17;
				x ^= x << 5;
			}
			expected[id] = x;
		}
		memset(source.contents, int(round + 1), COPY_BYTES);
		for (unsigned order = 0; order < 2; ++order) {
			const unsigned mode = order ^ (round & 1);
			context.SetConservativeBarriersForTesting(mode == 0);
			Check(context.BeginCommands(&error), error);
			context.ResetMetrics();
			context.Keep({output, source, destination, readback, pipeline});
			if (profile) {
				[counters invalidateCounterRange:NSMakeRange(0, 34)];
				[context.Commands() writeTimestampIntoHeap:counters atIndex:0];
			}
			const double begin = time_now_d();
			auto input = context.Upload(&seed, sizeof(seed), &error, 4);
			Check(bool(input), error);
			for (unsigned pass = 0; pass < 16; ++pass) {
				auto compute = context.ComputeEncoder(context.Commands(), "bench compute", true);
				[compute setComputePipelineState:pipeline];
				context.SetBuffer(compute, input.buffer, input.offset, 0);
				context.SetBuffer(compute, output, 0, 1);
				context.UseResource(compute, output, (MTLStages)0, MTLStageDispatch);
				[compute dispatchThreadgroups:MTLSizeMake(VERTICES / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
				context.EndEncoder(compute);
				if (profile) {
					[context.Commands() writeTimestampIntoHeap:counters atIndex:pass * 2 + 1];
				}
				auto blit = context.BlitEncoder(context.Commands(), "bench blit", true);
				context.UseResource(blit, source, MTLStageBlit, (MTLStages)0);
				context.UseResource(blit, destination, (MTLStages)0, MTLStageBlit);
				[blit copyFromBuffer:source sourceOffset:0 toBuffer:destination destinationOffset:0 size:COPY_BYTES];
				context.EndEncoder(blit);
				if (profile) {
					[context.Commands() writeTimestampIntoHeap:counters atIndex:pass * 2 + 2];
				}
			}
			auto capture = context.BlitEncoder(context.Commands(), "bench blit", true);
			context.UseResource(capture, destination, MTLStageBlit, (MTLStages)0);
			context.UseResource(capture, readback, (MTLStages)0, MTLStageBlit);
			[capture copyFromBuffer:destination sourceOffset:0 toBuffer:readback destinationOffset:0 size:COPY_BYTES];
			context.EndEncoder(capture);
			if (profile) {
				[context.Commands() writeTimestampIntoHeap:counters atIndex:33];
			}
			const double encoded = time_now_d();
			Check(context.SubmitCommands(true, &error), error);
			Check(memcmp(output.contents, expected.data(), expected.size() * sizeof(uint32_t)) == 0, "dependency benchmark compute contents");
			Check(memcmp(source.contents, readback.contents, COPY_BYTES) == 0, "dependency benchmark copy contents");
			double computeMs = 0.0, blitMs = 0.0;
			if (profile) {
				NSData *data = [counters resolveCounterRange:NSMakeRange(0, 34)];
				Check(data.length == 34 * sizeof(MTL4TimestampHeapEntry), "dependency timing counter data");
				const auto *timestamps = (const MTL4TimestampHeapEntry *)data.bytes;
				for (unsigned pass = 0; pass < 33; ++pass) {
					Check(timestamps[pass].timestamp && timestamps[pass + 1].timestamp >= timestamps[pass].timestamp, "dependency timing timestamp order");
					const double elapsed = (timestamps[pass + 1].timestamp - timestamps[pass].timestamp) * (1000.0 / context.TimestampFrequency());
					(pass < 32 && pass % 2 == 0 ? computeMs : blitMs) += elapsed;
				}
			}
			Check(context.LastSubmissionGPUTimeMs() > 0.0, "dependency benchmark GPU timestamps");
			Check(!profile || (computeMs > 0.0 && blitMs > 0.0), "completion timestamps cover compute and copy work");
			const auto metrics = context.Metrics();
			Check(metrics.fullQueueBarriers == (mode ? 0 : 33), "dependency benchmark uses selected policy");
			barriers[mode] = metrics.queueBarriers;
			full[mode] = metrics.fullQueueBarriers;
			if (round >= 8) {
				samples[mode].push_back({context.LastSubmissionGPUTimeMs(), (encoded - begin) * 1000.0,
					metrics.completionWaitSeconds * 1000.0, computeMs, blitMs});
			}
		}
	}
	for (const auto &[name, member] : std::initializer_list<std::pair<const char *, double Sample::*>>{
		{"gpu", &Sample::gpu}, {"encode", &Sample::encode}, {"cpu-wait", &Sample::wait},
		{"compute-completion-interval-sum", &Sample::computePasses}, {"blit-completion-interval-sum", &Sample::blitPasses}}) {
		if (!profile && (member == &Sample::computePasses || member == &Sample::blitPasses)) {
			continue;
		}
		std::vector<double> sorted[2];
		for (unsigned mode = 0; mode < 2; ++mode) {
			for (const auto &sample : samples[mode]) {
				sorted[mode].push_back(sample.*member);
			}
			std::sort(sorted[mode].begin(), sorted[mode].end());
		}
		for (unsigned percentile : {50, 95, 99}) {
			const size_t index = (sorted[0].size() - 1) * percentile / 100;
			std::cout << "MEASURE " << measurement << " " << name << " percentile=" << percentile
				<< " conservative-ms=" << sorted[0][index] << " targeted-ms=" << sorted[1][index] << " samples=72\n";
		}
	}
	std::cout << "MEASURE " << measurement << " barriers conservative=" << barriers[0] << " targeted=" << barriers[1]
		<< " full-conservative=" << full[0] << " full-targeted=" << full[1] << "\n";
	std::cout << "PASS " << measurement << " timing: 80 alternating pairs, exact compute/copy outputs, GPU/CPU intervals separated\n";
}

class CacheProbe : public TextureCacheMetal {
public:
	using TextureCacheMetal::TextureCacheMetal;
	using MetalTextureCacheCommon::DecodeTextureLevel;
	using MetalTextureCacheCommon::cache_;
	using MetalTextureCacheCommon::secondCache_;
	using MetalTextureCacheCommon::LoadTextureLevel;
	using MetalTextureCacheCommon::GetSamplingParams;
	using MetalTextureCacheCommon::ApplyTextureFinish;
	using MetalTextureCacheCommon::replacer_;
	void Palette() {
		for (int i = 0; i < 512; ++i) {
			clutBufRaw_[i] = clutBufConverted_[i] = 0xFF000000u | (i & 255) * 0x010101;
		}
		clutBuf_ = clutBufConverted_;
	}
	// Hash tests exercise ApplyTexture without allocating an unrelated GPU texture.
	void BuildTexture(MetalTexCacheEntry *) override {}
};

static void ReplacementUpdate(CacheProbe &cache) {
	const Path pack(std::filesystem::current_path().string());
	std::vector<u8> rgba(8 * 8 * 4);
	for (size_t i = 0; i < rgba.size(); ++i) {
		rgba[i] = (u8)(i * 17 + 23);
	}
	Check(pngSave(pack / "replacement-base.png", rgba.data(), 8, 8, 4), "replacement fixture PNG");
	std::vector<u8> mip(4 * 4 * 4, 0x5A);
	Check(pngSave(pack / "replacement-mip.png", mip.data(), 4, 4, 4), "replacement fixture mip");
	DirectoryReader directory(pack, true);
	MetalReplacementDesc desc{};
	desc.w = desc.h = 8;
	desc.newW = desc.newH = 8;
	desc.basePath = pack;
	desc.filenames = {"REPLACEMENT-BASE.PNG", "replacement-mip.png"};
	MetalReplacedTexture replaced(&directory, desc);
	g_threadManager.Init(1, 1);
	Check(replaced.Poll(5.0) && replaced.State() == ReplacementState::ACTIVE, "case-insensitive Metal replacement load");
	Check(replaced.NumLevels() == 2, "replacement mip chain loaded");
	g_threadManager.Teardown();
	// Upload copies must work without dispatching tasks to the worker pool.
	for (int pitch : {32, 48}) {
		std::vector<u8> out(pitch * 8, 0xA5);
		Check(replaced.CopyLevelTo(0, out.data(), out.size(), pitch), "caller-thread replacement copy");
		for (int y = 0; y < 8; ++y) {
			Check(memcmp(out.data() + y * pitch, rgba.data() + y * 32, 32) == 0, "replacement copy pixels");
		}
	}
	MetalTexCacheEntry entry{};
	entry.format = GE_TFMT_8888;
	entry.dim = 0x0303;
	entry.replacedTexture = &replaced;
	MetalBuildTexturePlan plan{};
	plan.doReplace = true;
	plan.replaced = &replaced;
	plan.scaleFactor = 1;
	for (int sourceLevel : {0, 1, 3, 7}) {
		plan.baseLevelSrc = sourceLevel;
		std::vector<u8> out(rgba.size(), 0);
		cache.LoadTextureLevel(entry, out.data(), out.size(), 32, plan, sourceLevel, Draw::DataFormat::R8G8B8A8_UNORM, TexDecodeFlags{});
		Check(out == rgba, "fake mipmap uses destination replacement level zero");
	}
	const bool oldReplace = g_Config.bReplaceTextures;
	g_Config.bReplaceTextures = true;
	std::ofstream("textures.ini") << "[options]\nversion=1\nhash=quick\n[hashes]\n";
	std::string packError;
	Check(cache.replacer_.LoadPackForTesting(pack, &packError), packError);
	Check(cache.replacer_.Enabled(), "replacement sampler branch enabled");
	const u32 oldFilter = gstate.texfilter;
	const u32 oldLevel = gstate.texlevel;
	gstate.texfilter = 0;
	gstate.texlevel = 0;
	Check(cache.GetSamplingParams(1, &entry, false, false).mipEnable, "multi-level replacement enables mip filtering");
	entry.status |= MetalTexStatus::NO_MIPS;
	Check(!cache.GetSamplingParams(0, &entry, false, false).mipEnable, "single uploaded level must not force replacement mip filtering");
	gstate.texfilter = oldFilter;
	gstate.texlevel = oldLevel;
	g_Config.bReplaceTextures = oldReplace;
	cache.cache_.clear();
	cache.secondCache_.clear();
	for (int i = 0; i < 4; ++i) {
		auto item = std::make_unique<MetalTexCacheEntry>();
		item->addr = 0x08810000 + i * 0x100;
		item->status = (MetalTexStatus)0;
		item->hostMemorySize = (i + 1) * 1024;
		cache.cache_[item->CacheKey()] = std::move(item);
	}
	auto secondary = std::make_unique<MetalTexCacheEntry>();
	secondary->status = (MetalTexStatus)0;
	auto secondaryPtr = secondary.get();
	cache.secondCache_[1] = std::move(secondary);
	cache.RecheckAllAfterLoad();
	Check(cache.cache_.size() == 4 && cache.secondCache_[1].get() == secondaryPtr, "savestate load retains both texture caches");
	for (const auto &[key, item] : cache.cache_) {
		Check(bool(item->status & MetalTexStatus::HASH_RECHECK), "retained texture must rehash after load");
		Check(item->EstimateTexMemoryUsage() == item->hostMemorySize, "host-size estimate retained after load");
	}
	Check(!bool(secondaryPtr->status & MetalTexStatus::HASH_RECHECK), "secondary hash-keyed cache stays valid");
	cache.cache_.clear();
	cache.secondCache_.clear();
	std::filesystem::remove("replacement-base.png");
	std::filesystem::remove("replacement-mip.png");
	std::filesystem::remove("textures.ini");
	std::cout << "PASS replacement-update: case-insensitive pack, caller copies, four fake mip levels, single-mip filtering, savestate rehash retention\n";
}

static void CurveLineEdges() {
	int cases = 0;
	for (int patchWidth : {1, 2, 3, 5}) {
		for (int patches : {1, 2, 3}) {
			const int columns = patchWidth * patches;
			const int rows = 3;
			std::vector<u16> indices(columns * rows * 6);
			int count = 0;
			Spline::BuildIndex(indices.data(), count, columns, rows, patchWidth, GE_PATCHPRIM_LINES);
			Check(count == rows * (columns * 4 + patches * 2), "line patch index count");
			std::map<std::pair<u16, u16>, int> edges;
			for (int i = 0; i < count; i += 2) {
				u16 a = indices[i], b = indices[i + 1];
				Check(a < (columns + 1) * (rows + 1) && b < (columns + 1) * (rows + 1), "curve index bounds");
				++edges[{std::min(a, b), std::max(a, b)}];
			}
			for (int y = 0; y < rows; ++y) {
				for (int x = 0; x <= columns; ++x) {
					u16 a = y * (columns + 1) + x, b = a + columns + 1;
					const int expected = x > 0 && x < columns && x % patchWidth == 0 ? 2 : 1;
					Check(edges[{a, b}] == expected, "shared quad edge once, inter-patch boundary twice");
				}
			}
			++cases;
		}
	}
	std::cout << "PASS curve-line-edges " << cases << " patch layouts; shared edges and inter-patch boundaries\n";
}

// Observe the matrix that a batch sees at the actual command dispatch boundary.
class BatchProbe : public DrawEngineMetal {
public:
	using DrawEngineMetal::DrawEngineMetal;
	std::vector<std::array<u32, 4>> matrices;
	void Flush() override {
		matrices.push_back({gstate.dithmtx[0], gstate.dithmtx[1], gstate.dithmtx[2], gstate.dithmtx[3]});
	}
};

class CommandProbe : public GPUCommonHWMetal {
public:
	BatchProbe batch;
	CommandProbe(Draw::DrawContext *draw) : GPUCommonHWMetal(nullptr, draw), batch(draw) {
		framebufferManager_ = new FramebufferManagerMetal(draw);
		textureCache_ = new CacheProbe(draw, framebufferManager_->GetDraw2D());
		static_cast<CacheProbe *>(textureCache_)->SetFramebufferManager(framebufferManager_);
		framebufferManager_->SetDrawEngine(&batch);
		drawEngineCommon_ = &batch;
	}
	~CommandProbe() override { framebufferManager_->SetDrawEngine(nullptr); }
	void GetStats(StringWriter &) override {}
	CacheProbe &Cache() { return *static_cast<CacheProbe *>(textureCache_); }
	void Write(u32 op, bool fast) {
		if (fast) {
			DisplayList list{};
			list.pc = 0x08800000;
			*reinterpret_cast<u32 *>(Memory::GetPointerWriteUnchecked(list.pc)) = op;
			downcount = 1;
			FastRunLoop(list);
		} else {
			u32 diff = gstate.cmdmem[op >> 24] ^ op;
			PreExecuteOp(op, diff);
			gstate.cmdmem[op >> 24] = op;
			ExecuteOp(op, diff);
		}
	}
};

static void Dither(CommandProbe &gpu) {
	int cases = 0;
	for (bool fast : {false, true}) {
		for (int row = 0; row < 4; ++row) {
			gpu.batch.matrices.clear();
			const u32 first = ((GE_CMD_DITH0 + row) << 24) | 0x0123;
			const u32 second = ((GE_CMD_DITH0 + row) << 24) | 0x7654;
			gstate.dithmtx[row] = first;
			gpu.Write(first, fast);
			Check(gpu.batch.matrices.empty(), "unchanged DITH must not flush");
			gpu.Write(second, fast);
			Check(gpu.batch.matrices.size() == 1 && gpu.batch.matrices[0][row] == first,
				"old batch must retain its DITH row before the register changes");
			gpu.batch.Flush();
			Check(gpu.batch.matrices.back()[row] == second, "next batch must see the new DITH row");
			++cases;
		}
	}
	std::cout << "PASS dither-command-boundaries " << cases << " (both loops, all rows, unchanged values)\n";
}

static void DepthMirrors(CacheProbe &cache) {
	int cases = 0;
	for (auto notification : {NOTIFY_FB_CREATED, NOTIFY_FB_UPDATED}) {
		VirtualFramebuffer fb{};
		fb.fb_address = 0x04080000;
		fb.z_address = 0x04010000;
		fb.z_stride = 512;
		fb.height = 32;
		const u32 length = 512 * 16 * 2;
		cache.cache_.clear();
		for (u32 mirror : {0u, 0x200000u, 0x400000u, 0x600000u}) {
			for (int offset : {-1, 0, 1, (int)length - 1, (int)length}) {
				for (u32 hash : {0u, 0x12345678u, 0xFFFFFFFFu}) {
					auto entry = std::make_unique<MetalTexCacheEntry>();
					entry->addr = (fb.z_address | mirror) + offset;
					entry->format = GE_TFMT_CLUT8;
					entry->dim = 0x0808;
					entry->cluthash = hash;
					entry->status = (MetalTexStatus)0;
					u64 key = entry->CacheKey();
					cache.cache_[key] = std::move(entry);
				}
			}
		}
		cache.NotifyFramebuffer(&fb, notification);
		for (auto &item : cache.cache_) {
			const auto &entry = *item.second;
			const u32 mirror = entry.addr & 0x600000;
			const u32 physical = entry.addr & ~0x600000;
			bool expected = (mirror == 0x200000 || mirror == 0x600000) &&
				physical >= fb.z_address && physical < fb.z_address + length;
			Check(bool(entry.status & MetalTexStatus::FRAMEBUFFER_OVERLAP) == expected,
				"depth mirror range/CLUT boundary " + std::to_string(entry.addr));
			++cases;
		}
		cache.cache_.clear();
	}
	std::cout << "PASS depth-mirror-cache-ranges " << cases << "\n";
}

// Independent byte-address oracle: each swizzle tile is 16 bytes by 8 rows.
static size_t SourceByteOffset(int xByte, int y, int rowBytes, bool swizzled) {
	if (!swizzled) {
		return y * rowBytes + xByte;
	}
	return (y / 8) * rowBytes * 8 + (xByte / 16) * 128 + (y % 8) * 16 + xByte % 16;
}

static void Swizzle(CacheProbe &cache) {
	cache.Palette();
	gstate.clutformat = GE_CMODE_32BIT_ABGR8888 | (255 << 8);
	int cases = 0;
	for (u32 base : {0x08800000u, 0x04000000u, 0x04200000u, 0x04400000u, 0x04600000u}) {
		for (bool raw : {false, true}) {
			const bool effective = raw != (base >= 0x04000000 && base < 0x04800000 && (base & 0x200000));
			gstate.texmode = raw ? 1 : 0;
			for (auto format : {GE_TFMT_CLUT4, GE_TFMT_CLUT8, GE_TFMT_CLUT16, GE_TFMT_CLUT32}) {
				const int bits = 4 << (format - GE_TFMT_CLUT4);
				for (int heightExp : {0, 2, 3, 4}) {
					for (int stride : {16, 32, 64}) {
						for (bool indicesOnly : {false, true}) {
							if (indicesOnly && bits > 8) {
								continue;
							}
							const int width = 32, height = 1 << heightExp;
							const int pixelBytes = indicesOnly ? 1 : 4;
							gstate.texsize[0] = 5 | (heightExp << 8);
							u8 *src = Memory::GetPointerWriteUnchecked(base);
							for (int i = 0; i < 8192; ++i) {
								src[i] = (i * 73 + (i >> 4) * 19 + 29) & 255;
							}
							std::vector<u8> output(width * height * pixelBytes + 64, 0xCD);
							cache.DecodeTextureLevel(output.data(), width * pixelBytes, format, GE_CMODE_32BIT_ABGR8888,
								base, 0, stride, indicesOnly ? TexDecodeFlags::TO_CLUT8 : TexDecodeFlags::EXPAND32);
							for (int y = 0; y < height; ++y) {
								for (int x = 0; x < width; ++x) {
									u8 index = src[SourceByteOffset(x * bits / 8, y, stride * bits / 8, effective)];
									if (bits == 4) {
										index = (index >> ((x & 1) * 4)) & 15;
									}
									const int dest = (y * width + x) * pixelBytes;
									Check(output[dest] == index, "indexed decode/layout at " + std::to_string(base) +
										" format=" + std::to_string(bits) + " raw=" + std::to_string(raw));
									if (!indicesOnly) {
										Check(output[dest + 1] == index && output[dest + 2] == index && output[dest + 3] == 255, "palette result");
									}
								}
							}
							for (size_t i = width * height * pixelBytes; i < output.size(); ++i) {
								Check(output[i] == 0xCD, "decode output canary");
							}
							++cases;
						}
					}
				}
			}
		}
	}
	std::cout << "PASS indexed-effective-swizzle " << cases << "\n";
}

static void HashLayout(CacheProbe &cache) {
	int cases = 0;
	gstate_c.SetUseFlags(0);
	gstate.texformat = GE_TFMT_8888;
	gstate.texsize[0] = 4;
	gstate.texlevel = 0;
	for (u32 addr : {0x08800000u, 0x04000000u, 0x04200000u, 0x04600000u}) {
		for (bool raw : {false, true}) {
			gstate.texmode = raw;
			gstate.texaddr[0] = addr & 0xFFFFFF;
			gstate.texbufwidth[0] = ((addr >> 8) & 0x0F0000) | 16;
			auto src = Memory::GetPointerWriteUnchecked(addr);
			memset(src, 0x15, 1024);
			cache.cache_.clear();
			cache.ApplyTexture(false);
			Check(cache.cache_.size() == 1, "hash fixture creates one entry");
			u32 hash = cache.cache_.begin()->second->fullhash;
			// Byte 256 is visible on row zero only in the swizzled layout.
			src[256] ^= 0xFF;
			cache.cache_.clear();
			cache.ApplyTexture(false);
			bool effective = raw != (Memory::IsVRAMAddress(addr) && (addr & 0x200000));
			Check((hash != cache.cache_.begin()->second->fullhash) == effective, "hash follows effective mirror layout");
			++cases;
		}
	}
	cache.cache_.clear();
	std::cout << "PASS effective-layout-hash " << cases << "\n";
}

static void DecodeBounds(CacheProbe &cache) {
	cache.Palette();
	gstate.clutformat = GE_CMODE_32BIT_ABGR8888 | (255 << 8);
	int cases = 0;
	for (u32 end : {0x08000000u + Memory::g_MemorySize, 0x04800000u}) {
		for (bool raw : {false, true}) {
			for (int stride : {16, 32, 64}) {
				for (auto format : {GE_TFMT_CLUT4, GE_TFMT_CLUT8, GE_TFMT_CLUT16, GE_TFMT_CLUT32}) {
					const int bits = 4 << (format - GE_TFMT_CLUT4);
					for (int available : {32, 128, 512}) {
						u32 addr = end - available;
						gstate.texmode = raw;
						gstate.texsize[0] = 5 | (2 << 8);
						memset(Memory::GetPointerWriteUnchecked(addr), 0x11, available);
						std::vector<u32> output(32 * 4 + 16, 0xCDCDCDCD);
						cache.DecodeTextureLevel((u8 *)output.data(), 32 * 4, format, GE_CMODE_32BIT_ABGR8888,
							addr, 0, stride, TexDecodeFlags::EXPAND32);
						bool effective = raw != (Memory::IsVRAMAddress(addr) && (addr & 0x200000));
						int checked = 0;
						for (int y = 0; y < 4; ++y) {
							for (int x = 0; x < 32; ++x) {
								if (output[y * 32 + x] != 0xCDCDCDCD) {
									Check(SourceByteOffset((x * bits + bits - 1) / 8, y, stride * bits / 8, effective) < available,
										"decoder must not consume bytes beyond the mapped range");
									Check(output[y * 32 + x] == (bits == 4 ? 0xFF010101u : 0xFF111111u), "bounded palette result");
									++checked;
								}
							}
						}
						if (stride == 16 && SourceByteOffset(32 * bits / 8 - 1, 3, stride * bits / 8, effective) < (size_t)available) {
							Check(checked > 0, "bounded test reached decode");
						}
						for (size_t i = 32 * 4; i < output.size(); ++i) {
							Check(output[i] == 0xCDCDCDCD, "bounded output canary");
						}
						++cases;
					}
				}
			}
		}
	}
	std::cout << "PASS decode-mapped-boundary " << cases << "\n";
}

static void Multipass(Metal::RenderContext &context) {
	std::string error;
	int cases = 0;
	for (int scale : {2, 4}) {
		MetalTextureShaderInfo info;
		info.name = "Internal dependent dispatch test";
		info.scaleFactor = scale;
		info.computeShaderFiles = scale == 2 ?
			std::vector<Path>{Path("metal-tests/first.comp"), Path("metal-tests/horizontal.comp"), Path("metal-tests/copy.comp")} :
			std::vector<Path>{Path("metal-tests/first.comp"), Path("metal-tests/vertical.comp"), Path("metal-tests/horizontal.comp"), Path("metal-tests/copy.comp")};
		TextureScalerMetal scaler;
		Check(scaler.Configure(context, info, &error), error);
		for (int iteration = 0; iteration < 8; ++iteration) {
			for (auto size : {std::pair{1, 1}, {7, 13}, {32, 8}, {63, 31}}) {
				int w = size.first, h = size.second;
				std::vector<u32> input(w * h);
				for (int i = 0; i < w * h; ++i) {
					input[i] = (i + iteration * 3) & 15;
				}
				if (!context.Commands()) {
					Check(context.BeginCommands(&error), error);
				}
				auto upload = context.Upload(input.data(), input.size() * sizeof(u32), &error);
				Draw::AutoRef<Metal::Texture> output;
				output.reset(scaler.Scale(context, upload, w, h, 1, &error));
				Check(output != nullptr, error);
				std::vector<u32> pixels(w * h * scale * scale);
				Check(Metal::Readback(context, output->Native(), Draw::Aspect::COLOR_BIT, 0, 0, w * scale, h * scale,
					Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), w * scale, &error), error);
				for (int y = 0; y < h * scale; ++y) {
					for (int x = 0; x < w * scale; ++x) {
						int expected = (int)std::round((input[(y / scale) * w + x / scale] / 32.0 + (scale == 2 ? 2 : 4) / 32.0) * 255.0);
						u32 pixel = pixels[y * w * scale + x];
						Check(std::abs(int(pixel & 255) - expected) <= 1 && (pixel >> 24) == 255, "dependent scaling dispatch output");
					}
				}
				++cases;
			}
		}
	}
	std::cout << "PASS native-multipass-scaling " << cases << "\n";
}

static void LibraryReuse() {
	Metal::RenderContext context;
	std::string error;
	Check(context.Init(&error, 2), error);
	Metal::CompiledShader shader;
	shader.source = R"(#include <metal_stdlib>
using namespace metal;
kernel void first(device uint *output [[buffer(0)]], uint i [[thread_position_in_grid]]) { output[i] = 17 + i; }
kernel void second(device uint *output [[buffer(0)]], uint i [[thread_position_in_grid]]) { output[i] = 93 + i; }
)";
	auto verify = [&](id<MTLFunction> function, uint32_t expected) {
		NSError *nativeError = nil;
		auto pipeline = context.CreateComputePipeline(function, &nativeError);
		Check(pipeline != nil, nativeError ? nativeError.localizedDescription.UTF8String : "library fixture pipeline");
		auto output = [context.Device() newBufferWithLength:8 * sizeof(uint32_t) options:MTLResourceStorageModeShared];
		Check(output != nil, "library fixture buffer");
		Check(context.BeginCommands(&error), error);
		context.Keep({pipeline, output});
		auto encoder = context.ComputeEncoder(context.Commands(), "library reuse", true);
		[encoder setComputePipelineState:pipeline];
		context.SetBuffer(encoder, output, 0, 0);
		context.UseResource(encoder, output, (MTLStages)0, MTLStageDispatch);
		[encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(8, 1, 1)];
		context.EndEncoder(encoder);
		Check(context.SubmitCommands(true, &error), error);
		const auto *values = (const uint32_t *)output.contents;
		for (uint32_t i = 0; i < 8; ++i) {
			Check(values[i] == expected + i, "library reuse selected the wrong entry point/source");
		}
	};
	shader.entryPoint = "first";
	id<MTLFunction> first = context.CreateShader(shader, "first caller", &error);
	Check(first != nil, error);
	verify(first, 17);
	shader.entryPoint = "second";
	id<MTLFunction> second = context.CreateShader(shader, "second caller", &error);
	Check(second != nil, error);
	verify(second, 93);
	std::cout << "MEASURE library-reuse exact-source two-entrypoints native-compiles=" << context.Metrics().libraryCompiles << " expected=1 GPU-values=16-exact\n";
	Check(context.Metrics().libraryCompiles == 1, "identical MSL library compiled again for a different caller/entry point");
	Check(context.Metrics().libraryCacheHits == 1, "library cache did not report reuse");
	const std::string original = shader.source;
	shader.entryPoint = "missing";
	Check(!context.CreateShader(shader, "missing entry", &error) && !error.empty(), "missing entry accepted");
	shader.entryPoint = "first";
	id<MTLFunction> repeated = context.CreateShader(shader, "another tag", &error);
	Check(repeated && context.Metrics().libraryCompiles == 1, "missing entry or debug tag poisoned a valid library");
	verify(repeated, 17);
	shader.source.replace(shader.source.find("17 + i"), 6, "61 + i");
	id<MTLFunction> changed = context.CreateShader(shader, "changed source", &error);
	Check(changed && context.Metrics().libraryCompiles == 2, "different MSL source aliased a cached library");
	verify(changed, 61);
	shader.source = "this is not valid MSL";
	for (unsigned i = 0; i < 2; ++i) {
		Check(!context.CreateShader(shader, "invalid source", &error) && !error.empty(), "invalid MSL accepted");
	}
	Check(context.Metrics().libraryCompiles == 4, "failed compilation was cached");
	shader.source = std::string(1, char(0xff));
	Check(!context.CreateShader(shader, "invalid UTF-8", &error) && !error.empty(), "invalid source encoding accepted");
	Check(context.Metrics().libraryCompiles == 4, "invalid UTF-8 reached the compiler");
	shader.source = original;
	shader.entryPoint = std::string(1, char(0xff));
	Check(!context.CreateShader(shader, "invalid entry encoding", &error) && !error.empty(), "invalid entry encoding accepted");
	shader.entryPoint = "first";
	{
		Metal::RenderContext other;
		Check(other.Init(&error, 2), error);
		Check(other.CreateShader(shader, "separate context", &error) != nil, error);
		Check(other.Metrics().libraryCompiles == 1, "library cache crossed renderer ownership");
	}
	std::vector<id<MTLFunction>> retained;
	for (unsigned i = 0; i < Metal::RenderContext::MAX_LIBRARY_CACHE_ENTRIES + 2; ++i) {
		shader.source = original + "// variant " + std::to_string(i) + "\n";
		id<MTLFunction> function = context.CreateShader(shader, "bounded cache", &error);
		Check(function != nil, error);
		retained.push_back(function);
		const auto metrics = context.Metrics();
		Check(metrics.libraryCacheEntries <= Metal::RenderContext::MAX_LIBRARY_CACHE_ENTRIES &&
			metrics.libraryCacheBytes <= Metal::RenderContext::MAX_LIBRARY_CACHE_BYTES, "library key cache exceeded its bounds");
	}
	Check(context.Metrics().libraryCacheEvictions > 0, "entry budget was not exercised");
	// Eviction removes a lookup key, never a live function's native library.
	verify(first, 17);
	for (unsigned i = 0; i < 3; ++i) {
		shader.source = original + "/*" + std::string(Metal::RenderContext::MAX_LIBRARY_CACHE_BYTES / 2, ' ') + "*/\n//" + std::to_string(i);
		id<MTLFunction> function = context.CreateShader(shader, "text budget", &error);
		Check(function != nil, error);
		retained.push_back(function);
		Check(context.Metrics().libraryCacheBytes <= Metal::RenderContext::MAX_LIBRARY_CACHE_BYTES, "text budget was not enforced");
	}
	const auto beforeLarge = context.Metrics();
	shader.source = original + "/*" + std::string(Metal::RenderContext::MAX_LIBRARY_CACHE_BYTES, ' ') + "*/";
	for (unsigned i = 0; i < 2; ++i) {
		id<MTLFunction> function = context.CreateShader(shader, "oversized key", &error);
		Check(function != nil, error);
		retained.push_back(function);
	}
	Check(context.Metrics().libraryCompiles == beforeLarge.libraryCompiles + 2 &&
		context.Metrics().libraryCacheEntries == beforeLarge.libraryCacheEntries &&
		context.Metrics().libraryCacheBytes == beforeLarge.libraryCacheBytes, "oversized source entered the bounded index");
	verify(retained.back(), 17);
	context.ResetMetrics();
	Check(context.Metrics().libraryCacheBytes == beforeLarge.libraryCacheBytes, "metric reset cleared the library cache");
	std::cout << "PASS library-reuse entrypoints/source/encoding/failures/context isolation; entry/text bounds and live evicted functions; 48 exact GPU values\n";
}

static void CacheParseGate() {
	const auto oldConfig = g_Config;
	const auto oldCached = gstate_c;
	struct Restore {
		std::function<void()> restore;
		~Restore() { restore(); }
	} restore{[&] { g_Config = oldConfig; gstate_c = oldCached; }};
	g_Config.bShaderCache = true;
	std::string error;
	std::unique_ptr<Draw::DrawContext> builder(Draw::T3DCreateMetalContext(&error));
	Check(builder != nullptr, error);
	ShaderManagerMetal shaders(builder.get());
	VShaderID vertex;
	vertex.clear();
	FShaderID fragment;
	fragment.clear();
	Check(shaders.GetVertexShaderFromID(vertex, &error) && shaders.GetFragmentShaderFromID(fragment, &error), error);
	const Path file("parse-gate.metalshadercache");
	shaders.SaveCache(file);
	Check(VerifyShaderCacheChecksum(file), "parse gate fixture save");
	std::ifstream input(file.ToString(), std::ios::binary);
	std::string body((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
	Check(body.size() > 64, "parse gate fixture size");
	body.resize(body.size() - 8);
	FILE *readerFile = File::OpenCFile(file, "rb");
	Check(readerFile && fseek(readerFile, 56, SEEK_SET) == 0, "parse gate header");
	Metal::TranslationCache::Reader reader(readerFile, body.size() - 56);
	Metal::TranslationCache::Record record;
	Check(reader.Value(&vertex) && Metal::TranslationCache::Read(reader, &record), "parse gate first record");
	const size_t fragmentOffset = ftell(readerFile);
	fclose(readerFile);
	std::array<std::string, 6> malformed{body.substr(0, fragmentOffset + sizeof(fragment) + 2), body + "x", body, body, body, body};
	fragment.set_invalid();
	memcpy(malformed[2].data() + fragmentOffset, &fragment, sizeof(fragment));
	vertex.set_invalid();
	memcpy(malformed[3].data() + 56, &vertex, sizeof(vertex));
	const uint32_t excessive = 8193, maximum = 8192;
	memcpy(malformed[4].data() + 32, &excessive, sizeof(excessive));
	memcpy(malformed[5].data() + 32, &maximum, sizeof(maximum));
	auto write = [&](const std::string &bytes) {
		std::ofstream output(file.ToString(), std::ios::binary | std::ios::trunc);
		output.write(bytes.data(), bytes.size());
		output.close();
		Check(AppendShaderCacheChecksum(file) && VerifyShaderCacheChecksum(file), "parse gate valid checksum");
	};
	unsigned compilerWork = 0;
	for (size_t i = 0; i < malformed.size(); ++i) {
		write(malformed[i]);
		// A separate renderer keeps this test independent of any live library cache.
		std::unique_ptr<Draw::DrawContext> draw(Draw::T3DCreateMetalContext(&error));
		Check(draw != nullptr, error);
		auto manager = (Metal::RenderManager *)draw->GetNativeObject(Draw::NativeObject::RENDER_MANAGER);
		ShaderManagerMetal sample(draw.get());
		manager->Context().ResetMetrics();
		Check(!sample.LoadCache(file), "malformed shader cache accepted");
		const auto native = manager->Context().Metrics();
		const auto metrics = sample.Metrics();
		std::cout << "MEASURE cache-parse-gate case=" << i << " generated=" << metrics.generated << " translated=" << metrics.translated << " native-compiles=" << native.libraryCompiles << '\n';
		compilerWork += metrics.generated != 0 || metrics.translated != 0 || native.libraryCompiles != 0;
		Check(sample.GetNumVertexShaders() == 0 && sample.GetNumFragmentShaders() == 0, "partial malformed cache retained");
	}
	Check(compilerWork == 0, "malformed cache reached the shader generator/compiler before validation finished");
	vertex.clear();
	fragment.clear();
	const auto *oldVS = shaders.GetVertexShaderFromID(vertex, &error);
	const auto *oldFS = shaders.GetFragmentShaderFromID(fragment, &error);
	const auto generation = shaders.CacheGeneration();
	const auto generated = shaders.Metrics().generated;
	for (const auto &bytes : malformed) {
		write(bytes);
		Check(!shaders.LoadCache(file), "malformed warm cache accepted");
		Check(shaders.CacheGeneration() == generation && shaders.Metrics().generated == generated &&
			shaders.GetVertexShaderFromID(vertex, &error) == oldVS && shaders.GetFragmentShaderFromID(fragment, &error) == oldFS,
			"rejected disk payload discarded valid live shaders");
	}
	write(body);
	auto builderManager = (Metal::RenderManager *)builder->GetNativeObject(Draw::NativeObject::RENDER_MANAGER);
	const auto beforeReload = builderManager->Context().Metrics();
	ShaderManagerMetal sameRenderer(builder.get());
	Check(sameRenderer.LoadCache(file), "valid same-renderer cache reload");
	const auto afterReload = builderManager->Context().Metrics();
	Check(sameRenderer.Metrics().translationCacheHits == 2 && afterReload.libraryCompiles == beforeReload.libraryCompiles &&
		afterReload.libraryCacheHits == beforeReload.libraryCacheHits + 2, "GE cache reload did not reuse both live native libraries");
	std::unique_ptr<Draw::DrawContext> cold(Draw::T3DCreateMetalContext(&error));
	Check(cold != nullptr, error);
	ShaderManagerMetal loaded(cold.get());
	Check(loaded.LoadCache(file) && loaded.Metrics().translationCacheHits == 2 && loaded.Metrics().translated == 0,
		"valid full payload no longer reuses translations");
	// A fully parsed payload can still fail native compilation. Partial shader
	// loads must be discarded in that case, as before the parse gate.
	std::string invalidMSL = body;
	uint32_t glslBytes;
	memcpy(&glslBytes, body.data() + fragmentOffset + sizeof(fragment) + 8, sizeof(glslBytes));
	const size_t mslOffset = fragmentOffset + sizeof(fragment) + 8 + 4 + glslBytes + 4;
	Check(mslOffset < body.size(), "fragment MSL offset");
	invalidMSL[mslOffset] = '@';
	write(invalidMSL);
	ShaderManagerMetal failed(cold.get());
	Check(!failed.LoadCache(file) && failed.Metrics().generated == 2 &&
		failed.GetNumVertexShaders() == 0 && failed.GetNumFragmentShaders() == 0, "failed native preload retained a partial shader set");
	File::Delete(file);
	std::cout << "PASS cache-parse-gate six checksum-valid malformed payloads cold/warm; no compiler work or live shader loss; valid cold/live-library reload and native failure rollback\n";
}

static void CacheIntegrity(Draw::DrawContext *draw, Metal::RenderManager *manager) {
	g_Config.bShaderCache = true;
	ShaderManagerMetal shaders(draw);
	PipelineManagerMetal pipelines(manager);
	VShaderID vertex;
	vertex.clear();
	FShaderID fragment;
	fragment.clear();
	std::string error;
	Check(shaders.GetVertexShaderFromID(vertex, &error), error);
	Check(shaders.GetFragmentShaderFromID(fragment, &error), error);
	Check(shaders.Metrics().generated == 2 && shaders.Metrics().translated == 2, "shader phase call counters");
	Check(shaders.Metrics().generationSeconds > 0.0 && shaders.Metrics().frontendSeconds > 0.0 &&
		shaders.Metrics().phases.glslSeconds > 0.0 && shaders.Metrics().phases.spirvSeconds > 0.0 &&
		shaders.Metrics().phases.mslSeconds > 0.0, "shader phase timers");
	Check(shaders.GetVertexShaderFromID(vertex, &error) && shaders.GetFragmentShaderFromID(fragment, &error), error);
	Check(shaders.Metrics().generated == 2 && shaders.Metrics().translated == 2, "cache hits counted as shader compilation");
	std::cout << "PASS shader-phase-metrics generator/GLSL/SPIR-V/MSL; memory hits do not compile\n";
	Check(!pipelines.PipelinesPending(), "empty pipeline cache must not delay boot");
	MetalGEBlendState blend;
	for (int mask = 1; mask <= 15; ++mask) {
		blend.writeMask = (MTLColorWriteMask)mask;
		pipelines.Request(shaders, vertex, fragment, nullptr, blend, MTLPixelFormatRGBA8Unorm,
			MTLPixelFormatInvalid, &error);
		Check(error.empty(), error);
	}
	const double deadline = time_now_d() + 30.0;
	bool observedPending = false;
	while (pipelines.PipelinesPending()) {
		observedPending = true;
		Check(time_now_d() < deadline, "asynchronous pipeline warmup did not finish");
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	for (int mask = 1; mask <= 15; ++mask) {
		blend.writeMask = (MTLColorWriteMask)mask;
		Check(pipelines.GetOrCreate(shaders, vertex, fragment, nullptr, blend, MTLPixelFormatRGBA8Unorm,
			MTLPixelFormatInvalid, &error), error);
	}
	const auto metrics = pipelines.Metrics();
	Check(metrics.asyncCompiles == 15 && metrics.asyncSeconds > 0.0 && metrics.syncCompiles == 0,
		"async pipeline metrics must count completion once and exclude memory hits");
	{
		PipelineManagerMetal sync(manager);
		Check(sync.GetOrCreate(shaders, vertex, fragment, nullptr, blend, MTLPixelFormatRGBA8Unorm,
			MTLPixelFormatInvalid, &error), error);
		Check(sync.GetOrCreate(shaders, vertex, fragment, nullptr, blend, MTLPixelFormatRGBA8Unorm,
			MTLPixelFormatInvalid, &error), error);
		Check(sync.Metrics().syncCompiles == 1 && sync.Metrics().syncSeconds > 0.0 && sync.Metrics().asyncCompiles == 0,
			"sync pipeline metrics must exclude memory hits");
	}
	std::cout << "PASS pipeline-phase-metrics sync/async completion; memory hits do not compile\n";
	{
		std::vector<double> samples;
		const auto *expected = pipelines.GetOrCreate(shaders, vertex, fragment, nullptr, blend,
			MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, &error);
		Check(expected != nullptr, error);
		for (int sample = 0; sample < 80; ++sample) {
			const double start = time_now_d();
			for (int draw = 0; draw < 10000; ++draw) {
				const auto *pipeline = pipelines.GetOrCreate(shaders, vertex, fragment, nullptr, blend,
					MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, &error);
				Check(pipeline == expected, "repeated pipeline lookup changed identity");
			}
			if (sample >= 8) {
				samples.push_back((time_now_d() - start) * 1000.0);
			}
		}
		std::sort(samples.begin(), samples.end());
		for (double quantile : {0.5, 0.95, 0.99}) {
			const size_t index = (size_t)std::ceil(quantile * samples.size()) - 1;
			std::cout << "MEASURE pipeline-lookup p=" << quantile << " elapsed-ms=" << samples[index]
				<< " lookups-per-sample=10000 samples=" << samples.size() << " cached-pipelines=15\n";
		}
	}
	int rejected = 0;
	for (bool shaderFile : {true, false}) {
		const Path filename(shaderFile ? "integrity.metalshadercache" : "integrity.metalpipelinecache");
		auto save = [&] {
			if (shaderFile) shaders.SaveCache(filename);
			else pipelines.SaveCache(filename);
		};
		auto load = [&] {
			ShaderManagerMetal restoredShaders(draw);
			if (shaderFile) {
				if (!restoredShaders.LoadCache(filename)) {
					return false;
				}
				Check(restoredShaders.Metrics().translationCacheHits == 2 && restoredShaders.Metrics().translated == 0,
					"warm shader cache repeated GLSL/SPIR-V/MSL compilation");
				const auto *oldVS = shaders.GetVertexShaderFromID(vertex, &error);
				const auto *newVS = restoredShaders.GetVertexShaderFromID(vertex, &error);
				const auto *oldFS = shaders.GetFragmentShaderFromID(fragment, &error);
				const auto *newFS = restoredShaders.GetFragmentShaderFromID(fragment, &error);
				Check(oldVS && newVS && oldFS && newFS && oldVS->source == newVS->source && oldFS->source == newFS->source &&
					oldVS->compiled.source == newVS->compiled.source && oldFS->compiled.source == newFS->compiled.source,
					"cached GLSL/MSL changed");
				return true;
			}
			PipelineManagerMetal restoredPipelines(manager);
			if (!restoredPipelines.LoadCache(filename, restoredShaders)) return false;
			const double until = time_now_d() + 30.0;
			while (restoredPipelines.PipelinesPending()) {
				Check(time_now_d() < until, "restored pipeline warmup did not finish");
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			Check(restoredPipelines.GetNumPipelines() == 15, "pipeline cache lost entries");
			for (int mask = 1; mask <= 15; ++mask) {
				MetalGEBlendState restoredBlend;
				restoredBlend.writeMask = (MTLColorWriteMask)mask;
				Check(restoredPipelines.GetOrCreate(restoredShaders, vertex, fragment, nullptr,
					restoredBlend, MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, &error), error);
			}
			return true;
		};
		save();
		Check(VerifyShaderCacheChecksum(filename) && load(), "valid cache roundtrip");
		if (shaderFile) {
			std::vector<double> cold, warm;
			uint64_t peakRSS = 0;
			for (int iteration = 0; iteration < 80; ++iteration) {
				for (int phase = 0; phase < 2; ++phase) {
					const bool cached = ((phase + iteration) & 1) != 0;
					@autoreleasepool {
						ShaderManagerMetal sample(draw);
						const double start = time_now_d();
						if (cached) {
							Check(sample.LoadCache(filename), "warm timing cache load");
						} else {
							Check(sample.GetVertexShaderFromID(vertex, &error) && sample.GetFragmentShaderFromID(fragment, &error), error);
						}
						const double elapsed = (time_now_d() - start) * 1000.0;
						Check(sample.Metrics().translated == (cached ? 0 : 2), "timing path did not execute");
						if (iteration >= 8) {
							(cached ? warm : cold).push_back(elapsed);
						}
					}
					mach_task_basic_info_data_t info{};
					mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
					Check(task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS, "resident memory measurement");
					peakRSS = std::max<uint64_t>(peakRSS, info.resident_size);
				}
			}
			std::sort(cold.begin(), cold.end());
			std::sort(warm.begin(), warm.end());
			for (double quantile : {0.5, 0.95, 0.99}) {
				const size_t index = (size_t)std::ceil(quantile * cold.size()) - 1;
				std::cout << "MEASURE translation-cache two-shader-load p=" << quantile << " uncached-ms=" << cold[index]
					<< " cached-ms=" << warm[index] << " samples=" << cold.size() << "\n";
			}
			std::cout << "MEASURE translation-cache file-bytes=" << File::GetFileSize(filename) << " combined-test-peak-rss=" << peakRSS << "\n";
		}

		std::ifstream file(filename.c_str(), std::ios::binary);
		std::string original((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
		file.close();
		auto write = [&](const std::string &data) {
			std::ofstream out(filename.c_str(), std::ios::binary | std::ios::trunc);
			out.write(data.data(), data.size());
			out.close();
			Check(out.good(), "write cache fixture");
		};
		// Every byte, including shader IDs/pipeline keys and the checksum trailer.
		for (size_t i = 0; i < original.size(); ++i) {
			std::string corrupt = original;
			corrupt[i] ^= 0x80;
			write(corrupt);
			Check(!load(), "damaged cache was accepted at byte " + std::to_string(i));
			++rejected;
		}
		for (size_t size : {size_t(0), size_t(7), original.size() / 2, original.size() - 1}) {
			write(original.substr(0, size));
			Check(!load(), "truncated cache was accepted");
			++rejected;
		}
		std::string oldVersion = original.substr(0, original.size() - 8);
		oldVersion[4] ^= 0x40;
		write(oldVersion);
		Check(AppendShaderCacheChecksum(filename), "checksum for format mismatch fixture");
		Check(!load(), "incompatible version with valid checksum was accepted");
		++rejected;
		if (!shaderFile) {
			auto previousVersion = original.substr(0, original.size() - 8);
			uint32_t version = 6;
			memcpy(previousVersion.data() + 4, &version, sizeof(version));
			write(previousVersion);
			Check(AppendShaderCacheChecksum(filename) && !load(), "old depth-dependent pipeline cache accepted");
			auto reservedDepth = original.substr(0, original.size() - 8);
			const uint64_t depth = MTLPixelFormatDepth32Float_Stencil8;
			memcpy(reservedDepth.data() + 112 + 3 * sizeof(uint64_t), &depth, sizeof(depth));
			write(reservedDepth);
			Check(AppendShaderCacheChecksum(filename) && !load(), "noncanonical depth cache key accepted");
			rejected += 2;
		}
		if (shaderFile) {
			// V5: 56-byte header (including image UUID), then ID, options, source length.
			for (size_t offset : {size_t(40), size_t(72)}) {
				auto invalid = original.substr(0, original.size() - 8);
				invalid[offset] ^= 0x80;
				if (offset == 72) {
					memset(invalid.data() + offset, 0xFF, 4);
				}
				write(invalid);
				Check(AppendShaderCacheChecksum(filename), "checksum for metadata fixture");
				Check(!load(), "compiler UUID/oversized source accepted");
				++rejected;
			}
			for (size_t offset : {size_t(64), size_t(76)}) {
				auto mismatched = original.substr(0, original.size() - 8);
				mismatched[offset] ^= 1;
				write(mismatched);
				Check(AppendShaderCacheChecksum(filename), "checksum for source/options mismatch");
				ShaderManagerMetal restored(draw);
				Check(restored.LoadCache(filename), "source/options mismatch did not rebuild");
				Check(restored.Metrics().translated == 1 && restored.Metrics().translationCacheHits == 1,
					"source/options mismatch reused stale translation");
			}
		}
		save();
		Check(VerifyShaderCacheChecksum(filename) && load(), "cache recovery after corruption");
		File::Delete(filename);
		File::Delete(filename.WithExtraExtension(".metal4archive"), true);
		for (int bank = 0; bank < 2; ++bank) {
			for (int i = 0; i < 8; ++i) {
				File::Delete(filename.WithExtraExtension(".metal4archive." + std::to_string(bank) + "." + std::to_string(i)), true);
			}
		}
	}
	pipelines.Clear();
	Check(!pipelines.PipelinesPending(), "cleared pipeline cache must not delay boot");
	g_Config.bShaderCache = false;
	std::cout << "PASS cache-integrity " << rejected << " rejected corrupt/truncated/version cases; valid roundtrip/recovery\n";
	std::cout << "PASS asynchronous-pipeline-warmup 15; observed pending=" << observedPending << "\n";
}

static std::vector<Path> ActiveArchiveFiles(const Path &filename) {
	std::ifstream input(filename.c_str(), std::ios::binary);
	uint32_t header[12]{};
	input.read((char *)header, sizeof(header));
	Check(input.good(), "read pipeline archive fixture header");
	if (header[1] == 5) {
		return {filename.WithExtraExtension(".metal4archive")};
	}
	Check(header[1] == 7 && header[9] <= 1 && header[10] <= 8, "pipeline archive bank/count");
	std::vector<Path> files;
	for (uint32_t i = 0; i < header[10]; ++i) {
		files.push_back(filename.WithExtraExtension(".metal4archive." + std::to_string(header[9]) + "." + std::to_string(i)));
	}
	return files;
}

static void ArchiveUnavailable(Draw::DrawContext *draw, Metal::RenderManager *manager) {
	const bool oldCache = g_Config.bShaderCache;
	g_Config.bShaderCache = true;
	const Path file("unavailable.metalpipelinecache");
	File::Delete(file, true);
	ShaderManagerMetal shaders(draw);
	PipelineManagerMetal pipelines(manager);
	VShaderID vertex;
	vertex.clear();
	FShaderID fragment;
	fragment.clear();
	std::string error;
	for (int count = 1; count <= 3; ++count) {
		MetalGEBlendState blend;
		blend.writeMask = (MTLColorWriteMask)count;
		Check(pipelines.GetOrCreate(shaders, vertex, fragment, nullptr, blend, MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, &error), error);
		pipelines.SaveCache(file);
		Check(pipelines.Metrics().archiveFailures == count, "expected native archive rejection did not occur");
		Check(VerifyShaderCacheChecksum(file) && ActiveArchiveFiles(file).empty(), "unavailable archive lost ID cache");
		ShaderManagerMetal restoredShaders(draw);
		PipelineManagerMetal restored(manager);
		Check(restored.LoadCache(file, restoredShaders), "unavailable archive IDs did not reload");
		for (int i = 1; i <= count; ++i) {
			blend.writeMask = (MTLColorWriteMask)i;
			Check(restored.GetOrCreate(restoredShaders, vertex, fragment, nullptr, blend, MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, &error), error);
		}
		Check(restored.GetNumPipelines() == count, "unavailable archive lost pipeline IDs");
	}
	File::Delete(file, true);
	g_Config.bShaderCache = oldCache;
	std::cout << "PASS archive-unavailable-fallback 3 growing ID saves and reloads; compiler recreates every PSO\n";
	std::cout << "SKIP progressive-native-archives in explicit unavailable-archive mode; separate normal/API runs required\n";
}

static void ProgressiveArchives(Draw::DrawContext *draw, Metal::RenderManager *manager) {
	const bool oldCache = g_Config.bShaderCache;
	g_Config.bShaderCache = true;
	const Path file("progressive.metalpipelinecache");
	auto cleanup = [&] {
		File::Delete(file, true);
		File::Delete(file.WithExtraExtension(".metal4archive"), true);
		for (int bank = 0; bank < 2; ++bank) {
			for (int i = 0; i < 8; ++i) {
				File::Delete(file.WithExtraExtension(".metal4archive." + std::to_string(bank) + "." + std::to_string(i)), true);
			}
		}
	};
	cleanup();
	ShaderManagerMetal shaders(draw);
	PipelineManagerMetal pipelines(manager);
	VShaderID vertex;
	vertex.clear();
	FShaderID fragment;
	fragment.clear();
	std::string error;
	auto get = [&](PipelineManagerMetal &cache, ShaderManagerMetal &shaderCache, int index) {
		MetalGEBlendState blend;
		blend.writeMask = (MTLColorWriteMask)(index % 15 + 1);
		const auto color = index < 15 ? MTLPixelFormatRGBA8Unorm : MTLPixelFormatBGRA8Unorm;
		const auto *pipeline = cache.GetOrCreate(shaderCache, vertex, fragment, nullptr, blend, color, MTLPixelFormatInvalid, &error);
		Check(pipeline && pipeline->state && pipeline->descriptor, error);
		return pipeline;
	};
	for (int step = 1; step <= 9; ++step) {
		for (int i = (step - 1) * 3; i < step * 3; ++i) {
			get(pipelines, shaders, i);
		}
		const auto before = pipelines.Metrics();
		pipelines.SaveCache(file);
		const auto after = pipelines.Metrics();
		Check(after.archivePipelines - before.archivePipelines == (step == 9 ? 27 : 3), "unchanged binaries recaptured outside compaction");
		pipelines.SaveCache(file);
		Check(pipelines.Metrics().archivePipelines == after.archivePipelines, "unchanged save rebuilt binaries");
		Check(VerifyShaderCacheChecksum(file), "progressive archive metadata checksum");
		const auto files = ActiveArchiveFiles(file);
		Check(!files.empty() && files.size() <= 8, "progressive archive count");
		NSMutableArray<id<MTL4Archive>> *archives = [NSMutableArray new];
		uint64_t bytes = 0;
		for (const auto &path : files) {
			NSError *nativeError = nil;
			auto archive = [manager->Context().Device() newArchiveWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]] error:&nativeError];
			Check(archive != nil, nativeError.localizedDescription.UTF8String ?: "open saved archive");
			[archives addObject:archive];
			bytes += File::GetFileSize(path);
		}
		for (int i = 0; i < step * 3; ++i) {
			const auto *pipeline = get(pipelines, shaders, i);
			bool found = false;
			for (id<MTL4Archive> archive in archives) {
				NSError *nativeError = nil;
				found |= [archive newRenderPipelineStateWithDescriptor:pipeline->descriptor error:&nativeError] != nil;
				if (found) break;
			}
			Check(found, "native archive lost pipeline " + std::to_string(i));
		}
		ShaderManagerMetal loadedShaders(draw);
		PipelineManagerMetal loaded(manager);
		Check(loaded.LoadCache(file, loadedShaders), "progressive archive load");
		const double deadline = time_now_d() + 30.0;
		while (loaded.PipelinesPending()) {
			Check(time_now_d() < deadline, "progressive archive warmup timeout");
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		Check(loaded.GetNumPipelines() == step * 3, "progressive archive lost IDs");
		for (int i = 0; i < step * 3; ++i) {
			get(loaded, loadedShaders, i);
		}
		std::cout << "MEASURE archive-save step=" << step << " pipelines=" << step * 3
			<< " captures=" << after.archivePipelines - before.archivePipelines
			<< " elapsed-ms=" << (after.archiveSeconds - before.archiveSeconds) * 1000.0
			<< " save-total-ms=" << (after.cacheSaveSeconds - before.cacheSaveSeconds) * 1000.0
			<< " files=" << files.size() << " bytes=" << bytes << "\n";
	}
	const auto readMetadata = [&] {
		std::ifstream input(file.c_str(), std::ios::binary);
		return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
	};
	const auto stableMetadata = readMetadata();
	const auto stableArchives = ActiveArchiveFiles(file);
	get(pipelines, shaders, 27);
	// Block metadata publication after the new native archive has been created.
	const auto blocked = std::filesystem::path(file.WithExtraExtension(".tmp").c_str());
	Check(!std::filesystem::exists(blocked), "stale metadata failure fixture");
	Check(std::filesystem::create_directory(blocked), "create metadata failure fixture");
	pipelines.SaveCache(file);
	std::filesystem::remove(blocked);
	Check(readMetadata() == stableMetadata && ActiveArchiveFiles(file) == stableArchives,
		"failed metadata save replaced last complete cache");
	for (const auto &path : stableArchives) {
		Check(File::Exists(path), "failed metadata save removed existing archive");
	}
	pipelines.SaveCache(file);
	Check(VerifyShaderCacheChecksum(file) && ActiveArchiveFiles(file).size() == 2, "save did not recover after metadata failure");
	// Reopen the saved cache, then remove/corrupt one part. IDs must still warm
	// through the compiler and the following save must restore a complete bank.
	for (bool corrupt : {false, true}) {
		const auto parts = ActiveArchiveFiles(file);
		if (corrupt) {
			std::ofstream output(parts.back().c_str(), std::ios::binary | std::ios::trunc);
			output << "invalid Metal archive";
		} else {
			Check(File::Delete(parts.back()), "remove archive failure fixture");
		}
		ShaderManagerMetal fallbackShaders(draw);
		PipelineManagerMetal fallback(manager);
		Check(fallback.LoadCache(file, fallbackShaders), "valid IDs rejected with missing/corrupt archive");
		const double deadline = time_now_d() + 30.0;
		while (fallback.PipelinesPending()) {
			Check(time_now_d() < deadline, "fallback archive warmup timeout");
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		Check(fallback.GetNumPipelines() == 28, "missing archive lost pipeline IDs");
		for (int i = 0; i < 28; ++i) {
			get(fallback, fallbackShaders, i);
		}
		fallback.SaveCache(file);
		const auto repaired = ActiveArchiveFiles(file);
		Check(repaired.size() == 1, "archive repair did not rebuild complete bank");
		NSError *nativeError = nil;
		auto archive = [manager->Context().Device() newArchiveWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:repaired[0].c_str()]] error:&nativeError];
		Check(archive != nil, "repaired native archive unavailable");
		for (int i = 0; i < 28; ++i) {
			Check([archive newRenderPipelineStateWithDescriptor:get(fallback, fallbackShaders, i)->descriptor error:&nativeError] != nil,
				"repair lost native binary");
		}
	}
	std::cout << "PASS archive-failure-recovery metadata failure preserves previous bank; missing/corrupt parts retain all 28 IDs and rebuild binaries\n";
	{
		ShaderManagerMetal resumedShaders(draw);
		PipelineManagerMetal resumed(manager);
		Check(resumed.LoadCache(file, resumedShaders), "resume incremental archives");
		const double deadline = time_now_d() + 30.0;
		while (resumed.PipelinesPending()) {
			Check(time_now_d() < deadline, "resumed archive warmup timeout");
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		get(resumed, resumedShaders, 28);
		resumed.SaveCache(file);
		Check(resumed.Metrics().archivePipelines == 1 && ActiveArchiveFiles(file).size() == 2,
			"restart save recaptured existing binaries");
		const auto stable = readMetadata();
		const auto parts = ActiveArchiveFiles(file);
		auto blockedArchive = std::filesystem::path(parts[0].c_str());
		blockedArchive.replace_extension(".2.tmp");
		Check(std::filesystem::create_directory(blockedArchive), "create archive failure fixture");
		std::ofstream(blockedArchive / "keep") << "prevent replacement of directory";
		get(resumed, resumedShaders, 29);
		resumed.SaveCache(file);
		Check(readMetadata() == stable && ActiveArchiveFiles(file) == parts, "archive write failure replaced valid cache");
		std::filesystem::remove(blockedArchive / "keep");
		std::filesystem::remove(blockedArchive);
		resumed.SaveCache(file);
		const auto complete = ActiveArchiveFiles(file);
		Check(complete.size() == 3, "archive write did not recover");
		NSMutableArray<id<MTL4Archive>> *archives = [NSMutableArray new];
		for (const auto &path : complete) {
			NSError *nativeError = nil;
			auto archive = [manager->Context().Device() newArchiveWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]] error:&nativeError];
			Check(archive != nil, "resumed native archive unavailable");
			[archives addObject:archive];
		}
		for (int i = 0; i < 30; ++i) {
			bool found = false;
			for (id<MTL4Archive> archive in archives) {
				NSError *nativeError = nil;
				found |= [archive newRenderPipelineStateWithDescriptor:get(resumed, resumedShaders, i)->descriptor error:&nativeError] != nil;
				if (found) break;
			}
			Check(found, "resumed incremental archives lost native binary");
		}
		ShaderManagerMetal finalShaders(draw);
		PipelineManagerMetal finalCache(manager);
		Check(finalCache.LoadCache(file, finalShaders) && finalCache.GetNumPipelines() == 30, "final archive IDs lost");
		for (int i = 0; i < 30; ++i) {
			get(finalCache, finalShaders, i);
		}
	}
	std::cout << "PASS archive-restart incremental append; archive write failure/recovery; all 30 native binaries and IDs preserved\n";
	cleanup();
	g_Config.bShaderCache = oldCache;
	std::cout << "PASS progressive-archives 9 saves; every native binary and ID reload preserved\n";
}

static void PipelineIdentity(Draw::DrawContext *draw, Metal::RenderManager *manager) {
	ShaderManagerMetal shaders(draw);
	PipelineManagerMetal pipelines(manager);
	VShaderID vertex;
	vertex.clear();
	FShaderID fragment;
	fragment.clear();
	MetalGEBlendState blend;
	std::string error;
	auto get = [&](VShaderID vs, FShaderID fs, const DecVtxFormat *layout, const MetalGEBlendState &state,
		MTLPixelFormat color = MTLPixelFormatRGBA8Unorm, MTLPixelFormat depth = MTLPixelFormatInvalid, int samples = 1) {
		const auto *pipeline = pipelines.GetOrCreate(shaders, vs, fs, layout, state, color, depth, &error, samples);
		Check(pipeline != nullptr, error);
		return pipeline;
	};
	const auto *base = get(vertex, fragment, nullptr, blend);
	Check(base == get(vertex, fragment, nullptr, blend), "identical pipeline identity");
	int variants = 0;
	for (int field = 0; field < 8; ++field) {
		auto changed = blend;
		switch (field) {
		case 0: changed.enabled = true; break;
		case 1: changed.srcColor = MTLBlendFactorSourceAlpha; break;
		case 2: changed.dstColor = MTLBlendFactorOne; break;
		case 3: changed.colorOp = MTLBlendOperationSubtract; break;
		case 4: changed.srcAlpha = MTLBlendFactorSourceAlpha; break;
		case 5: changed.dstAlpha = MTLBlendFactorOne; break;
		case 6: changed.alphaOp = MTLBlendOperationSubtract; break;
		case 7: changed.writeMask = MTLColorWriteMaskRed; break;
		}
		Check(get(vertex, fragment, nullptr, changed) != base, "blend key ignored field");
		Check(get(vertex, fragment, nullptr, blend) == base, "blend restoration lost pipeline");
		++variants;
	}
	Check(get(vertex, fragment, nullptr, blend, MTLPixelFormatBGRA8Unorm) != base, "target color ignored");
	const auto beforeDepth = pipelines.Metrics().syncCompiles;
	const auto *withDepth = get(vertex, fragment, nullptr, blend, MTLPixelFormatRGBA8Unorm, MTLPixelFormatDepth32Float_Stencil8);
	std::cout << "MEASURE depth-format-only additional-pipelines=" << pipelines.Metrics().syncCompiles - beforeDepth << '\n';
	Check(withDepth == base && pipelines.Metrics().syncCompiles == beforeDepth,
		"Metal 4 compiled an identical pipeline for a different depth attachment");
	Check(!pipelines.GetOrCreate(shaders, vertex, fragment, nullptr, blend, MTLPixelFormatRGBA8Unorm,
		MTLPixelFormatDepth16Unorm, &error), "unsupported depth bypassed validation");
	for (int samples : {2, 4}) {
		Check(get(vertex, fragment, nullptr, blend, MTLPixelFormatRGBA8Unorm, MTLPixelFormatInvalid, samples) != base, "samples ignored");
	}
	Check(!pipelines.GetOrCreate(shaders, vertex, fragment, nullptr, blend, MTLPixelFormatRGBA8Unorm,
		MTLPixelFormatInvalid, &error, 0), "invalid samples bypassed validation");
	Check(!pipelines.GetOrCreate(shaders, vertex, fragment, nullptr, blend, MTLPixelFormatInvalid,
		MTLPixelFormatInvalid, &error), "invalid target bypassed validation");
	vertex.SetBit(VS_BIT_USE_HW_TRANSFORM, true);
	DecVtxFormat layout{};
	layout.stride = 64;
	const auto *hardware = get(vertex, fragment, &layout, blend);
	Check(hardware != base && hardware->stride == 64, "hardware transform identity");
	Check(!pipelines.GetOrCreate(shaders, vertex, fragment, nullptr, blend, MTLPixelFormatRGBA8Unorm,
		MTLPixelFormatInvalid, &error), "missing hardware layout bypassed validation");
	for (int field = 0; field < 14; ++field) {
		auto changed = layout;
		switch (field) {
		case 0: changed.stride = 80; break;
		case 1: changed.posoff = 16; break;
		case 2: changed.uvfmt = DEC_FLOAT_2; break;
		case 3: changed.uvoff = 16; break;
		case 4: changed.c0fmt = DEC_U8_4; break;
		case 5: changed.c0off = 16; break;
		case 6: changed.c1fmt = DEC_U8_4; break;
		case 7: changed.c1off = 16; break;
		case 8: changed.nrmfmt = DEC_FLOAT_3; break;
		case 9: changed.nrmoff = 16; break;
		case 10: changed.w0fmt = DEC_FLOAT_4; break;
		case 11: changed.w0off = 16; break;
		case 12: changed.w1fmt = DEC_FLOAT_4; break;
		case 13: changed.w1off = 16; break;
		}
		Check(get(vertex, fragment, &changed, blend) != hardware, "decoded layout ignored field");
		Check(get(vertex, fragment, &layout, blend) == hardware, "layout restoration lost pipeline");
		++variants;
	}
	for (int reset = 0; reset < 4; ++reset) {
		const auto count = pipelines.Metrics().syncCompiles;
		switch (reset) {
		case 0: shaders.ClearShaders(); break;
		case 1: pipelines.Clear(); break;
		case 2:
			pipelines.DeviceLost();
			Check(!pipelines.GetOrCreate(shaders, vertex, fragment, &layout, blend, MTLPixelFormatRGBA8Unorm,
				MTLPixelFormatInvalid, &error), "lost device returned cached pipeline");
			pipelines.DeviceRestore(manager);
			break;
		case 3: g_Config.bVendorBugChecksEnabled = !g_Config.bVendorBugChecksEnabled; break;
		}
		get(vertex, fragment, &layout, blend);
		Check(pipelines.Metrics().syncCompiles == count + 1 && pipelines.GetNumPipelines() == 1,
			"pipeline identity survived invalidation");
	}
	g_Config.bVendorBugChecksEnabled = !g_Config.bVendorBugChecksEnabled;
	ShaderManagerMetal other(draw);
	const auto count = pipelines.Metrics().syncCompiles;
	Check(pipelines.GetOrCreate(other, vertex, fragment, &layout, blend, MTLPixelFormatRGBA8Unorm,
		MTLPixelFormatInvalid, &error), error);
	Check(pipelines.Metrics().syncCompiles == count + 1 && pipelines.GetNumPipelines() == 1,
		"pipeline identity survived shader owner change");
	std::cout << "PASS pipeline-identity " << variants << " blend/layout variants; targets, samples, transform, generation/environment/owner/device invalidation\n";
}

class AtlasProbe : public TextureCacheMetal {
public:
	using TextureCacheMetal::TextureCacheMetal;
	using TextureCacheMetal::BindTexture;
	using TextureCacheMetal::BuildTexture;
	using TextureCacheMetal::ReleaseTexture;
	using MetalTextureCacheCommon::cache_;
};

static void AtlasPressure(Draw::DrawContext *draw, Metal::RenderManager *manager) {
	const auto oldState = gstate;
	const auto oldCachedState = gstate_c;
	const bool oldScaling = g_Config.bTexHardwareScaling;
	const int oldScale = g_Config.iTexScalingLevel;
	const auto oldShader = g_Config.sTextureShaderName;
	GPUCommon *oldGPU = gpu;
	CommandProbe commands(draw);
	gpu = &commands;
	struct Cleanup {
		std::function<void()> restore;
		~Cleanup() { restore(); }
	} cleanup{[&] {
		gpu = oldGPU;
		gstate = oldState;
		gstate_c = oldCachedState;
		g_Config.bTexHardwareScaling = oldScaling;
		g_Config.iTexScalingLevel = oldScale;
		g_Config.sTextureShaderName = oldShader;
	}};
	g_Config.bTexHardwareScaling = true;
	g_Config.iTexScalingLevel = 2;
	g_Config.sTextureShaderName = "TexSpline36";
	gstate_c.SetUseFlags(0);
	gstate.texformat = GE_TFMT_8888;
	gstate.texsize[0] = 9 | (9 << 8);
	gstate.texmode = 0;
	gstate.texlevel = 0;
	AtlasProbe cache(draw, nullptr);
	cache.NotifyConfigChanged();
	std::array<MetalTexCacheEntry *, 2> entries;
	const std::array<u32, 2> colors{0xFF4080C0, 0xFFC08040};
	auto build = [&](int index, u32 color) {
		const u32 addr = 0x08800000 + index * 0x100000;
		gstate.texaddr[0] = addr & 0xFFFFFF;
		gstate.texbufwidth[0] = ((addr >> 8) & 0x0F0000) | 512;
		std::fill_n((u32 *)Memory::GetPointerWriteUnchecked(addr), 512 * 512, color);
		cache.BuildTexture(entries[index]);
		Check(entries[index]->texturePtr != nullptr, "atlas source build");
		cache.BindTexture(entries[index]);
		Check(cache.HasScaledTexture(), "atlas fixture did not reach hardware scaling");
	};
	for (int index = 0; index < 2; ++index) {
		auto entry = std::make_unique<MetalTexCacheEntry>();
		entry->addr = 0x08800000 + index * 0x100000;
		entry->format = GE_TFMT_8888;
		entry->dim = 9 | (9 << 8);
		entry->bufw = 512;
		entries[index] = entry.get();
		cache.cache_.emplace(entry->CacheKey(), std::move(entry));
		build(index, colors[index]);
	}
	std::string error;
	auto bind = [&](int index, std::array<int, 4> bounds, bool read, u32 expected) {
		if (!manager->Context().Commands()) {
			Check(manager->Context().BeginCommands(&error), error);
		}
		cache.BindTexture(entries[index]);
		TransformedVertex vertices[4]{};
		for (int i = 0; i < 4; ++i) {
			vertices[i].u = bounds[(i & 1) ? 2 : 0] / 512.0f;
			vertices[i].v = bounds[(i & 2) ? 3 : 1] / 512.0f;
		}
		TransformedVertex adjusted[4]{};
		Check(cache.BindSamplingRegion(vertices, adjusted, &bounds), "atlas crop bind");
		for (int i = 0; i < 4; ++i) {
			Check(adjusted[i].u == float(i & 1) && adjusted[i].v == float((i & 2) >> 1), "atlas UV remapping changed");
		}
		const auto metrics = cache.GetRegionMetrics();
		Check(metrics.count <= 256 && metrics.regionBytes <= 16 * 1024 * 1024, "atlas retained budget exceeded");
		if (read) {
			cache.RestoreGETextureBindings();
			auto texture = (__bridge id<MTLTexture>)(void *)draw->GetNativeObject(Draw::NativeObject::BOUND_TEXTURE0_IMAGEVIEW);
			Check(texture && texture.width == (bounds[2] - bounds[0]) * 2 && texture.height == (bounds[3] - bounds[1]) * 2, "scaled crop dimensions");
			u32 pixel = 0;
			Check(Metal::Readback(manager->Context(), texture, Draw::Aspect::COLOR_BIT, 0, 0, 1, 1,
				Draw::DataFormat::R8G8B8A8_UNORM, &pixel, 1, &error), error);
			for (int shift : {0, 8, 16, 24}) {
				Check(std::abs(int((pixel >> shift) & 255) - int((expected >> shift) & 255)) <= 1, "atlas crop pixels changed");
			}
		}
	};
	for (int iteration = 0; iteration < 1000; ++iteration) {
		const int cold = iteration / 2 + 8;
		bind(iteration & 1, {cold % 512, cold / 512, cold % 512 + 1, cold / 512 + 1}, false, 0);
		for (int hot = 0; hot < 16; ++hot) {
			bind(hot & 1, {hot / 2, 0, hot / 2 + 1, 1}, iteration == 999, colors[hot & 1]);
		}
	}
	const auto countPressure = cache.GetRegionMetrics();
	Check(countPressure.misses == 1016 && countPressure.hits == 15984 && countPressure.evictions == 760 &&
		countPressure.clears == 0 && countPressure.count == 256, "atlas hot set lost under count pressure");
	std::cout << "MEASURE atlas count-pressure hits/misses/evictions/clears=" << countPressure.hits << "/" << countPressure.misses
		<< "/" << countPressure.evictions << "/" << countPressure.clears << " entries=" << countPressure.count << " bytes=" << countPressure.regionBytes << "\n";
	for (int i = 0; i < 12; ++i) {
		bind(1, {0, i, 512, i + 384}, false, 0);
		bind(0, {0, 0, 512, 256}, i == 11, colors[0]);
	}
	const auto bytePressure = cache.GetRegionMetrics();
	Check(bytePressure.misses - countPressure.misses == 13 && bytePressure.clears == 0,
		"atlas hot region lost under byte pressure");
	std::cout << "MEASURE atlas byte-pressure additional-misses=" << bytePressure.misses - countPressure.misses
		<< " additional-clears=" << bytePressure.clears - countPressure.clears << " entries=" << bytePressure.count << " bytes=" << bytePressure.regionBytes << "\n";
	// Rebuild one source while other sources still have cached regions.
	build(0, 0xFF20A060);
	bind(0, {0, 0, 1, 1}, true, 0xFF20A060);
	bind(1, {0, 0, 1, 1}, true, colors[1]);
	cache.DeviceLost();
	const auto empty = cache.GetRegionMetrics();
	Check(empty.count == 0 && empty.regionBytes == 0 && empty.sourceBytes == 0, "atlas source removal accounting");
	cache.DeviceRestore(draw);
	std::cout << "PASS atlas-pressure 17000 count-pressure and 24 byte-pressure binds; UV/pixels, source rebuild, device loss\n";
}

static void UploadPools() {
	Metal::RenderContext context;
	std::string error;
	Check(context.Init(&error, 2), error);
	Check(context.BeginCommands(&error), error);
	for (size_t alignment : {size_t(0), size_t(3), size_t(512)}) {
		Check(!context.ReserveUpload(8, &error, alignment), "invalid upload alignment accepted");
	}
	for (size_t alignment : {size_t(256), size_t(4)}) {
		context.ResetMetrics();
		for (int i = 0; i < 4096; ++i) {
			auto upload = context.ReserveUpload(6, &error, alignment);
			Check(upload && upload.offset % alignment == 0, "index upload alignment");
		}
		auto metrics = context.Metrics();
		Check(metrics.uploadRequests == 4096 && metrics.uploadBytes == 24576, "upload counters");
		Check(alignment != 4 || metrics.uploadPadding <= 4096 * 2, "packed index padding");
		std::cout << "MEASURE index-upload alignment=" << alignment << " payload=" << metrics.uploadBytes
			<< " padding=" << metrics.uploadPadding << " capacity=" << metrics.uploadCapacity << "\n";
	}
	Check(context.SubmitCommands(true, &error), error);
	// Grow both in-flight slots, then keep them busy with small requests.
	std::vector<id<MTLBuffer>> outputs;
	for (int iteration = 0; iteration < 74; ++iteration) {
		Check(context.BeginCommands(&error), error);
		auto upload = context.ReserveUpload(iteration < 2 ? 8 * 1024 * 1024 : 8, &error, 4);
		Check((bool)upload, error);
		const uint64_t marker = 0x1234567800000000ULL | iteration;
		memcpy((uint8_t *)upload.buffer.contents + upload.offset, &marker, sizeof(marker));
		id<MTLBuffer> output = [context.Device() newBufferWithLength:8 options:MTLResourceStorageModeShared];
		for (int i = 0; i < (iteration < 2 ? 64 : 1); ++i) {
			auto encoder = context.BlitEncoder(context.Commands(), "Pool lifetime test");
			context.SetBuffer(encoder, upload.buffer, upload.offset, 0);
			context.Keep(output);
			[encoder copyFromBuffer:upload.buffer sourceOffset:upload.offset toBuffer:output destinationOffset:0 size:8];
			context.EndEncoder(encoder);
		}
		outputs.push_back(output);
		Check(context.SubmitCommands(false, &error), error);
		if (iteration == 1 || iteration == 73) {
			auto metrics = context.Metrics();
			std::cout << "MEASURE upload-pool iteration=" << iteration << " capacity=" << metrics.uploadCapacity
				<< " argument-tables=" << metrics.argumentCapacity << "\n";
		}
	}
	Check(context.WaitUntilIdle(&error), error);
	for (size_t i = 0; i < outputs.size(); ++i) {
		Check(*(const uint64_t *)outputs[i].contents == (0x1234567800000000ULL | i),
			"asynchronous upload slot reused before GPU completion");
	}
	Check(context.Metrics().uploadCapacity <= 2 * 1024 * 1024, "upload peak retained after sustained small use");
	Check(context.Metrics().argumentCapacity <= 32, "argument table peak retained after sustained small use");
	std::cout << "PASS upload-pool-alignment-lifetime-shrink\n";
}

static void LazyFrameCommands() {
	using namespace Draw;
	const int savedInflight = g_Config.iInflightFrames;
	for (int slots : {1, 2}) {
		g_Config.iInflightFrames = slots;
		std::string error;
		std::unique_ptr<DrawContext> draw(T3DCreateMetalContext(&error));
		Check(draw != nullptr, error);
		auto manager = (Metal::RenderManager *)draw->GetNativeObject(NativeObject::RENDER_MANAGER);
		auto &context = manager->Context();
		AutoRef<Framebuffer> fb(draw->CreateFramebuffer({8, 8, 1, 1, 0, true, "lazy frame test"}));
		Check(fb != nullptr, "lazy frame framebuffer");
		for (uint32_t frame = 0; frame < 16; ++frame) {
			const auto waits = context.Metrics().completionWaits;
			draw->BeginFrame(DebugFlags::NONE);
			Check(!context.Commands() && context.Metrics().completionWaits == waits,
				"frame start recycled GPU storage before its first use");
			const uint32_t color = 0xFF102030 + frame;
			draw->BindFramebufferAsRenderTarget(fb, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
				color, 0.375f, (uint8_t)frame, "lazy frame clear"}, "lazy frame clear");
			draw->EndFrame();
			std::array<uint32_t, 64> pixels;
			Check(draw->CopyFramebufferToMemory(fb, Aspect::COLOR_BIT, 0, 0, 8, 8,
				DataFormat::R8G8B8A8_UNORM, pixels.data(), 8, ReadbackMode::BLOCK, "lazy frame readback"),
				"lazy frame clear readback");
			Check(std::all_of(pixels.begin(), pixels.end(), [color](uint32_t pixel) { return pixel == color; }),
				"lazy frame lost a deferred clear after submission");
		}
	}
	g_Config.iInflightFrames = savedInflight;
	std::cout << "PASS lazy-frame commands: one/two slots, 32 clears/readbacks, no wait at frame start\n";
}

static void AlphaStencilFetch(Draw::DrawContext *draw, Metal::RenderManager *manager) {
	using namespace Draw;
	auto &context = manager->Context();
	int cases = 0;
	for (int scale : {1, 8}) {
		const int width = 257 * scale, height = 3 * scale;
		AutoRef<Framebuffer> framebuffer(draw->CreateFramebuffer({width, height, 1, 1, 0, true, "alpha-stencil test"}));
		AutoRef<Framebuffer> other(draw->CreateFramebuffer({8, 8, 1, 1, 0, true, "alpha-stencil other target"}));
		Check(framebuffer != nullptr, "alpha-stencil framebuffer");
		for (int format : {GE_FORMAT_5551, GE_FORMAT_4444, GE_FORMAT_8888}) {
			std::vector<uint32_t> source(width * height), colors(source.size());
			std::vector<float> depths(source.size());
			std::vector<uint8_t> stencils(source.size());
			for (size_t i = 0; i < source.size(); ++i) {
				uint32_t alpha = i & 255;
				if (format == GE_FORMAT_5551) alpha = alpha & 128 ? 255 : 0;
				if (format == GE_FORMAT_4444) alpha = (alpha >> 4) * 17;
				source[i] = (alpha << 24) | ((i * 1777) & 0xFFFFFF);
			}
			draw->BeginFrame(DebugFlags::NONE);
			draw->BindFramebufferAsRenderTarget(framebuffer, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
				0, 0.375f, 173, "alpha-stencil initial"}, "alpha-stencil initial");
			manager->EndRenderPass();
			auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
				width:width height:height mipmapped:NO];
			desc.storageMode = MTLStorageModeShared;
			desc.usage = MTLTextureUsageShaderRead;
			auto texture = [context.Device() newTextureWithDescriptor:desc];
			Check(texture != nil, "alpha-stencil source texture");
			[texture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0
				withBytes:source.data() bytesPerRow:width * 4];
			std::string error;
			Check(Metal::CopyImage(context, texture, 0, 0, static_cast<Metal::Framebuffer *>(framebuffer.ptr)->Color(),
				0, 0, width, height, &error), error);
			const bool separateTarget = format == GE_FORMAT_4444;
			if (separateTarget) {
				draw->BindFramebufferAsRenderTarget(other, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
					0x71335577, 0.625f, 91, "other target"}, "other target");
			} else {
				Check(manager->RenderEncoder() != nil, "alpha-stencil active pass");
			}
			const auto passes = context.Metrics().renderPasses;
			Check(T3DMetalWriteStencilFromAlpha(draw, framebuffer), "native alpha-stencil fetch");
			Check(separateTarget || context.Metrics().renderPasses == passes, "alpha-stencil fetch split the active pass");
			Check(draw->CopyFramebufferToMemory(framebuffer, Aspect::COLOR_BIT, 0, 0, width, height,
				DataFormat::R8G8B8A8_UNORM, colors.data(), width, ReadbackMode::BLOCK, "alpha-stencil color"), "alpha-stencil color readback");
			Check(draw->CopyFramebufferToMemory(framebuffer, Aspect::DEPTH_BIT, 0, 0, width, height,
				DataFormat::D32F, depths.data(), width, ReadbackMode::BLOCK, "alpha-stencil depth"), "alpha-stencil depth readback");
			Check(draw->CopyFramebufferToMemory(framebuffer, Aspect::STENCIL_BIT, 0, 0, width, height,
				DataFormat::S8, stencils.data(), width, ReadbackMode::BLOCK, "alpha-stencil stencil"), "alpha-stencil stencil readback");
			Check(colors == source, "alpha-stencil fetch changed color");
			for (size_t i = 0; i < source.size(); ++i) {
				Check(depths[i] == 0.375f && stencils[i] == (source[i] >> 24), "alpha-stencil fetch changed depth or missed alpha");
			}
			++cases;
		}
	}
	AutoRef<Framebuffer> msaa(draw->CreateFramebuffer({8, 8, 1, 1, 2, true, "alpha-stencil MSAA fallback"}));
	Check(msaa && !T3DMetalWriteStencilFromAlpha(draw, msaa), "MSAA must retain the resolved-alpha path");
	std::cout << "PASS alpha-stencil framebuffer fetch " << cases << " format/scale cases, exact RGBA/depth/stencil; MSAA fallback\n";
}

static void DeferredClears(Draw::DrawContext *draw, Metal::RenderManager *manager) {
	using namespace Draw;
	auto &context = manager->Context();
	constexpr int size = 8;
	const RenderPassInfo keep{RPAction::KEEP, RPAction::KEEP, RPAction::KEEP, 0, 1.0f, 0, "keep"};
	int cases = 0;
	for (int msaa : {0, 1, 2}) {
		if (![context.Device() supportsTextureSampleCount:1 << msaa]) {
			continue;
		}
		AutoRef<Framebuffer> a(draw->CreateFramebuffer({size, size, 1, 1, msaa, true, "clear source"}));
		AutoRef<Framebuffer> b(draw->CreateFramebuffer({size, size, 1, 1, 0, true, "copy target"}));
		Check(a && b, "clear test framebuffer allocation");
		auto verify = [&](Framebuffer *fb, uint32_t color, float depth, uint8_t stencil) {
			std::array<uint32_t, size * size> colors{};
			std::array<float, size * size> depths{};
			std::array<uint8_t, size * size> stencils{};
			Check(draw->CopyFramebufferToMemory(fb, Aspect::COLOR_BIT, 0, 0, size, size,
				DataFormat::R8G8B8A8_UNORM, colors.data(), size, ReadbackMode::BLOCK, "clear color"), "color readback");
			Check(draw->CopyFramebufferToMemory(fb, Aspect::DEPTH_BIT, 0, 0, size, size,
				DataFormat::D32F, depths.data(), size, ReadbackMode::BLOCK, "clear depth"), "depth readback");
			Check(draw->CopyFramebufferToMemory(fb, Aspect::STENCIL_BIT, 0, 0, size, size,
				DataFormat::S8, stencils.data(), size, ReadbackMode::BLOCK, "clear stencil"), "stencil readback");
			for (int i = 0; i < size * size; ++i) {
				Check(colors[i] == color && depths[i] == depth && stencils[i] == stencil,
					"clear attachment mismatch msaa=" + std::to_string(msaa));
			}
			++cases;
		};
		for (bool forceSplit : {true, false}) {
			manager->EndRenderPass();
			context.ResetMetrics();
			draw->BindFramebufferAsRenderTarget(a, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
				0xFF332211, 0.75f, 19, "initial clear"}, "initial clear");
			if (forceSplit) {
				manager->EndRenderPass();
			}
			draw->Clear(Aspect::COLOR_BIT, 0xFF665544, 0.0f, 0);
			if (forceSplit) {
				manager->EndRenderPass();
			}
			draw->Clear(Aspect::DEPTH_BIT, 0, 0.25f, 0);
			if (forceSplit) {
				manager->EndRenderPass();
			}
			draw->BindFramebufferAsRenderTarget(a, keep, "keep pending clear");
			draw->Clear(Aspect::STENCIL_BIT, 0, 0.0f, 43);
			draw->EndFrame();
			auto passes = context.Metrics().renderPasses;
			Check(passes == (forceSplit ? 4 : 1), "clear pass reduction");
			verify(a, 0xFF665544, 0.25f, 43);
			std::cout << "MEASURE clear-pass msaa=" << msaa << " force-split=" << forceSplit << " passes=" << passes << "\n";
		}
		// An actual consumer boundary must preserve the earlier values.
		draw->BindFramebufferAsRenderTarget(a, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
			0xFF998877, 0.5f, 71, "copy source"}, "copy source");
		draw->CopyFramebufferImage(a, 0, 0, 0, 0, b, 0, 0, 0, 0, size, size, 1,
			Aspect::COLOR_BIT | Aspect::DEPTH_BIT | Aspect::STENCIL_BIT, "consume pending clear");
		draw->Clear(Aspect::COLOR_BIT, 0xFFCCBBAA, 0.0f, 0);
		verify(b, 0xFF998877, 0.5f, 71);
		verify(a, 0xFFCCBBAA, 0.5f, 71);
		draw->Clear(Aspect::DEPTH_BIT, 0, 0.125f, 0);
		draw->BindFramebufferAsRenderTarget(b, keep, "target switch");
		verify(a, 0xFFCCBBAA, 0.125f, 71);
		// Opening the native GE encoder consumes the pending clear exactly once.
		draw->BindFramebufferAsRenderTarget(a, keep, "GE target");
		draw->Clear(Aspect::STENCIL_BIT, 0, 0, 96);
		Check(manager->RenderEncoder() != nil, "GE encoder after deferred clear");
		manager->EndRenderPass();
		verify(a, 0xFFCCBBAA, 0.125f, 96);
		if (msaa) {
			draw->Clear(Aspect::DEPTH_BIT, 0, 0.375f, 0);
			draw->BindFramebufferAsTexture(a, 0, Aspect::DEPTH_BIT, 0);
			auto *native = static_cast<Metal::Framebuffer *>(a.ptr);
			Check(!native->NeedsDepthStencilResolve(), "deferred clear must resolve before depth sampling");
			std::array<float, size * size> depths{};
			std::string error;
			Check(Metal::Readback(context, native->DepthStencil(), Aspect::DEPTH_BIT, 0, 0, size, size,
				DataFormat::D32F, depths.data(), size, &error), error);
			for (float depth : depths) {
				Check(depth == 0.375f, "sampled depth used pre-clear resolve");
			}
			++cases;
		}
	}
	std::cout << "PASS deferred-clear-boundaries " << cases << " attachment sets\n";
}

static Metal::Framebuffer *TransferTarget(Metal::RenderContext &context, int samples, int layers, bool depth, bool bgra) {
	std::string error;
	if (!bgra) {
		auto fb = Metal::Framebuffer::Create(context.Device(), {8, 8, 1, layers, samples, depth, "transfer cache test"}, &error);
		Check(fb != nullptr, error);
		return fb;
	}
	Check(samples == 0 && layers == 1, "BGRA fixture must be single-sample/single-layer");
	auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm width:8 height:8 mipmapped:NO];
	desc.storageMode = MTLStorageModePrivate;
	desc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
	auto color = [context.Device() newTextureWithDescriptor:desc];
	desc.pixelFormat = MTLPixelFormatDepth32Float_Stencil8;
	auto ds = depth ? [context.Device() newTextureWithDescriptor:desc] : nil;
	auto fb = Metal::Framebuffer::Wrap(color, ds, &error);
	Check(fb != nullptr, error);
	return fb;
}

static void ClearTransferTarget(Metal::RenderContext &context, Metal::Framebuffer *fb, bool source) {
	std::string error;
	if (!context.Commands()) {
		Check(context.BeginCommands(&error), error);
	}
	for (int layer = 0; layer < fb->Layers(); ++layer) {
		const uint32_t color = (source ? 0x71403020 : 0xE0806040) + layer * 0x00010101;
		auto pass = [MTL4RenderPassDescriptor new];
		fb->SetRenderAttachments(pass, layer);
		pass.colorAttachments[0].loadAction = MTLLoadActionClear;
		pass.colorAttachments[0].clearColor = MTLClearColorMake((color & 255) / 255.0, ((color >> 8) & 255) / 255.0,
			((color >> 16) & 255) / 255.0, (color >> 24) / 255.0);
		if (fb->DepthStencil()) {
			pass.depthAttachment.loadAction = pass.stencilAttachment.loadAction = MTLLoadActionClear;
			pass.depthAttachment.clearDepth = source ? 0.25 + layer * 0.125 : 0.75 + layer * 0.0625;
			pass.stencilAttachment.clearStencil = (source ? 173 : 91) + layer;
			if (fb->SampleCount() > 1) {
				// Match EndPass: resolve depth/stencil only before a single-sample consumer.
				pass.depthAttachment.resolveTexture = pass.stencilAttachment.resolveTexture = nil;
				pass.depthAttachment.storeAction = pass.stencilAttachment.storeAction = MTLStoreActionStore;
				fb->MarkDepthStencilUnresolved();
			}
		}
		auto encoder = context.RenderEncoder(context.Commands(), pass);
		Check(encoder != nil, "transfer fixture clear encoder");
		context.EndEncoder(encoder);
	}
}

static void VerifyTransferTarget(Metal::RenderContext &context, Metal::Framebuffer *fb, Draw::Aspect aspects, bool alphaStencil = false) {
	using namespace Draw;
	std::string error;
	for (int layer = 0; layer < fb->Layers(); ++layer) {
		std::array<uint32_t, 64> pixels;
		Check(Metal::Readback(context, fb->ColorLayer(layer), Aspect::COLOR_BIT, 0, 0, 8, 8,
			DataFormat::R8G8B8A8_UNORM, pixels.data(), 8, &error), error);
		const uint32_t color = ((aspects & Aspect::COLOR_BIT) ? 0x71403020 : 0xE0806040) + layer * 0x00010101;
		Check(std::all_of(pixels.begin(), pixels.end(), [&](uint32_t p) { return p == color; }), "transfer color/layer mismatch");
		if (fb->DepthStencil()) {
			Check(fb->ResolveDepthStencil(context, &error), error);
			std::array<float, 64> depths;
			std::array<uint8_t, 64> stencils;
			Check(Metal::Readback(context, fb->DepthStencilLayer(layer), Aspect::DEPTH_BIT, 0, 0, 8, 8,
				DataFormat::D32F, depths.data(), 8, &error), error);
			Check(Metal::Readback(context, fb->DepthStencilLayer(layer), Aspect::STENCIL_BIT, 0, 0, 8, 8,
				DataFormat::S8, stencils.data(), 8, &error), error);
			const float z = (aspects & Aspect::DEPTH_BIT) ? 0.25f + layer * 0.125f : 0.75f + layer * 0.0625f;
			const uint8_t stencil = alphaStencil ? color >> 24 : ((aspects & Aspect::STENCIL_BIT) ? 173 : 91) + layer;
			for (size_t i = 0; i < depths.size(); ++i) {
				if (depths[i] != z || stencils[i] != stencil) {
					std::cout << "MEASURE transfer mismatch samples=" << fb->SampleCount() << " layers=" << fb->Layers()
						<< " layer=" << layer << " aspects=" << (int)aspects << " pixel=" << i << " depth=" << depths[i]
						<< " expected=" << z << " stencil=" << (int)stencils[i] << " expected=" << (int)stencil << '\n';
					break;
				}
			}
			Check(std::all_of(depths.begin(), depths.end(), [&](float p) { return p == z; }) &&
				std::all_of(stencils.begin(), stencils.end(), [&](uint8_t p) { return p == stencil; }), "transfer depth/stencil/layer mismatch");
		}
	}
}

static void TransferPipelineIdentity() {
	using namespace Draw;
	int cases = 0;
	for (int samples : {0, 2}) {
		for (auto filter : {FB_BLIT_NEAREST, FB_BLIT_LINEAR}) {
			Metal::RenderContext context;
			std::string error;
			Check(context.Init(&error, 2), error);
			Metal::FramebufferCopy copy;
			AutoRef<Metal::Framebuffer> src, plain, depth;
			src.reset(TransferTarget(context, 0, 1, true, false));
			plain.reset(TransferTarget(context, samples, 1, false, false));
			depth.reset(TransferTarget(context, samples, 1, true, false));
			ClearTransferTarget(context, src, true);
			ClearTransferTarget(context, plain, false);
			ClearTransferTarget(context, depth, false);
			Check(copy.Blit(context, src, 0, 0, 8, 8, plain, 0, 0, 8, 8, Aspect::COLOR_BIT, filter, &error), error);
			VerifyTransferTarget(context, plain, Aspect::COLOR_BIT);
			const auto compiled = context.Metrics().libraryCompiles;
			const auto pipelines = copy.GetNumPipelines();
			Check(copy.Blit(context, src, 0, 0, 8, 8, depth, 0, 0, 8, 8, Aspect::COLOR_BIT, filter, &error), error);
			VerifyTransferTarget(context, depth, Aspect::COLOR_BIT);
			std::cout << "MEASURE transfer depth-only key samples=" << (1 << samples) << " filter=" << filter
				<< " additional-libraries=" << context.Metrics().libraryCompiles - compiled
				<< " additional-pipelines=" << copy.GetNumPipelines() - pipelines << '\n';
			Check(copy.GetNumPipelines() == pipelines && context.Metrics().libraryCompiles == compiled,
				"depth presence rebuilt an identical transfer pipeline or fragment library");
			++cases;
		}
	}
	std::cout << "PASS transfer pipeline identity: " << cases << " nearest/linear and 1x/MSAA4 cases; exact color, depth and stencil preserved\n";
}

static void UtilityShaderReuse(bool alphaStencil) {
	using namespace Draw;
	Metal::RenderContext context;
	std::string error;
	Check(context.Init(&error, 2), error);
	Metal::FramebufferCopy copy;
	AutoRef<Metal::Framebuffer> src;
	src.reset(TransferTarget(context, 0, 1, true, false));
	ClearTransferTarget(context, src, true);
	uint64_t compiled = 0;
	int cases = 0;
	for (int variant = 0; variant < (alphaStencil ? 2 : 4); ++variant) {
		AutoRef<Metal::Framebuffer> dst;
		dst.reset(TransferTarget(context, variant > 1 ? variant - 1 : 0, 1, true, variant == 1));
		ClearTransferTarget(context, dst, false);
		if (alphaStencil) {
			Check(copy.StencilFromAlpha(context, dst, &error), error);
			VerifyTransferTarget(context, dst, Aspect::NO_BIT, true);
		} else {
			Check(copy.Blit(context, src, 0, 0, 8, 8, dst, 0, 0, 8, 8, Aspect::COLOR_BIT, FB_BLIT_NEAREST, &error), error);
			VerifyTransferTarget(context, dst, Aspect::COLOR_BIT);
		}
		Check(copy.GetNumPipelines() == (size_t)variant + 1, "distinct format/sample targets shared one pipeline");
		if (variant == 0) {
			compiled = context.Metrics().libraryCompiles;
		} else {
			std::cout << "MEASURE utility shader reuse alpha=" << alphaStencil << " variant=" << variant
				<< " additional-libraries=" << context.Metrics().libraryCompiles - compiled << '\n';
			Check(context.Metrics().libraryCompiles == compiled, "format/sample changes rebuilt identical utility shader libraries");
		}
		++cases;
	}
	if (!alphaStencil) {
		// Distinguish nearest and linear in the same function cache, using a nonuniform source.
		std::array<uint32_t, 64> gradientPixels;
		for (int y = 0; y < 8; ++y) {
			for (int x = 0; x < 8; ++x) {
				gradientPixels[y * 8 + x] = 0xFF4D0000 | (y * 16 << 8) | x * 32;
			}
		}
		auto desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:8 height:8 mipmapped:NO];
		desc.storageMode = MTLStorageModeShared;
		desc.usage = MTLTextureUsageShaderRead;
		auto texture = [context.Device() newTextureWithDescriptor:desc];
		Check(texture != nil, "filter fixture texture");
		[texture replaceRegion:MTLRegionMake2D(0, 0, 8, 8) mipmapLevel:0 withBytes:gradientPixels.data() bytesPerRow:32];
		AutoRef<Metal::Framebuffer> gradient, output;
		gradient.reset(Metal::Framebuffer::Wrap(texture, nil, &error));
		output.reset(TransferTarget(context, 0, 1, true, false));
		Check(gradient != nullptr, error);
		for (auto filter : {FB_BLIT_NEAREST, FB_BLIT_LINEAR, FB_BLIT_NEAREST, FB_BLIT_LINEAR}) {
			ClearTransferTarget(context, output, false);
			Check(copy.Blit(context, gradient, 0, 0, 8, 8, output, 0, 0, 4, 4, Aspect::COLOR_BIT, filter, &error), error);
			std::array<uint32_t, 16> pixels;
			Check(Metal::Readback(context, output->Color(), Aspect::COLOR_BIT, 0, 0, 4, 4,
				DataFormat::R8G8B8A8_UNORM, pixels.data(), 4, &error), error);
			for (int y = 0; y < 4; ++y) {
				for (int x = 0; x < 4; ++x) {
					const uint32_t expected = 0xFF4D0000 | ((y * 32 + (filter == FB_BLIT_LINEAR ? 8 : 16)) << 8) |
						(x * 64 + (filter == FB_BLIT_LINEAR ? 16 : 32));
					for (int shift : {0, 8, 16, 24}) {
						Check(std::abs(int((pixels[y * 4 + x] >> shift) & 255) - int((expected >> shift) & 255)) <= 1,
							"nearest/linear shader variants were confused");
					}
				}
			}
			++cases;
		}
		uint64_t warmed = 0;
		size_t warmedPipelines = 0;
		for (int repeat = 0; repeat < 2; ++repeat) {
			for (int layers : {1, 2}) {
				for (auto samples : {std::pair{0, 0}, {0, 2}, {2, 2}, {2, 0}}) {
					AutoRef<Metal::Framebuffer> source, target;
					source.reset(TransferTarget(context, samples.first, layers, true, false));
					target.reset(TransferTarget(context, samples.second, layers, true, false));
					for (int bits = 1; bits <= 7; ++bits) {
						const auto aspects = ((bits & 1) ? Aspect::COLOR_BIT : Aspect::NO_BIT) |
							((bits & 2) ? Aspect::DEPTH_BIT : Aspect::NO_BIT) | ((bits & 4) ? Aspect::STENCIL_BIT : Aspect::NO_BIT);
						ClearTransferTarget(context, source, true);
						ClearTransferTarget(context, target, false);
						if ((aspects & (Aspect::DEPTH_BIT | Aspect::STENCIL_BIT)) && samples.first != samples.second) {
							Check(source->ResolveDepthStencil(context, &error), error);
						}
						Check(copy.Blit(context, source, 0, 0, 8, 8, target, 0, 0, 8, 8, aspects, FB_BLIT_NEAREST, &error), error);
						if (samples.second && (aspects & (Aspect::DEPTH_BIT | Aspect::STENCIL_BIT))) {
							target->MarkDepthStencilResolved();
						}
						VerifyTransferTarget(context, target, aspects);
						++cases;
					}
				}
			}
			if (!repeat) {
				warmed = context.Metrics().libraryCompiles;
				warmedPipelines = copy.GetNumPipelines();
			} else {
				Check(copy.GetNumPipelines() == warmedPipelines && context.Metrics().libraryCompiles == warmed,
					"warm transfer variants rebuilt pipelines or libraries");
			}
		}
	}
	std::cout << "PASS utility shader reuse alpha=" << alphaStencil << ": " << cases
		<< " format/sample/aspect/layer cases with checked GPU output\n";
}

static void CopyPlaneReuse() {
	using namespace Draw;
	std::string error;
	const int previousInflight = g_Config.iInflightFrames;
	g_Config.iInflightFrames = 2;
	std::unique_ptr<DrawContext> draw(T3DCreateMetalContext(&error));
	g_Config.iInflightFrames = previousInflight;
	Check(draw != nullptr, error);
	auto *manager = (Metal::RenderManager *)draw->GetNativeObject(NativeObject::RENDER_MANAGER);
	auto &context = manager->Context();
	constexpr int size = 8, planeBytes = 256 * size;
	AutoRef<Framebuffer> src(draw->CreateFramebuffer({size, size, 1, 2, 0, true, "copy plane source"}));
	AutoRef<Framebuffer> dst(draw->CreateFramebuffer({size, size, 1, 2, 0, true, "copy plane target"}));
	Check(src && dst, "copy plane framebuffer allocation");
	auto clear = [&](Framebuffer *fb, float depth, uint8_t stencil) {
		draw->BindFramebufferAsRenderTarget(fb, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR,
			0xFF123456, depth, stencil, "copy plane clear"}, "copy plane clear");
	};
	auto capture = [&](id<MTLBuffer> output, int phase) {
		manager->EndRenderPass();
		auto encoder = context.BlitEncoder(context.Commands(), "copy plane test capture");
		Check(encoder != nil, "copy plane capture encoder");
		auto texture = static_cast<Metal::Framebuffer *>(dst.ptr)->DepthStencil();
		context.Keep({texture, output});
		for (int layer = 0; layer < 2; ++layer) {
			for (int aspect = 0; aspect < 2; ++aspect) {
				[encoder copyFromTexture:texture sourceSlice:layer sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
					sourceSize:MTLSizeMake(size, size, 1) toBuffer:output destinationOffset:(phase * 4 + layer * 2 + aspect) * planeBytes
					destinationBytesPerRow:256 destinationBytesPerImage:planeBytes
					options:aspect ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionDepthFromDepthStencil];
			}
		}
		context.EndEncoder(encoder);
	};
	auto verify = [&](id<MTLBuffer> output, int phase, auto expected) {
		for (int layer = 0; layer < 2; ++layer) {
			for (int y = 0; y < size; ++y) {
				const auto *depth = (const float *)((const uint8_t *)output.contents + (phase * 4 + layer * 2) * planeBytes + y * 256);
				const auto *stencil = (const uint8_t *)output.contents + (phase * 4 + layer * 2 + 1) * planeBytes + y * 256;
				for (int x = 0; x < size; ++x) {
					const auto value = expected(x);
					Check(depth[x] == value.first && stencil[x] == value.second, "copy plane data/counterpart/layer mismatch");
				}
			}
		}
	};
	std::vector<id<MTLBuffer>> outputs;
	for (int iteration = 0; iteration < 74; ++iteration) {
		draw->BeginFrame(DebugFlags::NONE);
		if (iteration < 2) {
			Check(context.BeginCommands(&error), error);
			Check(context.AcquireCopyPlane(4 * 1024 * 1024, &error) != nil, error);
		}
		clear(src, iteration & 1 ? 0.75f : 0.25f, iteration + 1);
		clear(dst, 0.5f, 170);
		id<MTLBuffer> output = [context.Device() newBufferWithLength:8 * planeBytes options:MTLResourceStorageModeShared];
		Check(output != nil, "copy plane output allocation");
		draw->CopyFramebufferImage(src, 0, 0, 0, 0, dst, 0, 0, 0, 0, size, size, 1, Aspect::DEPTH_BIT, "depth only");
		capture(output, 0);
		draw->CopyFramebufferImage(src, 0, 0, 0, 0, dst, 0, 0, 0, 0, size, size, 1, Aspect::STENCIL_BIT, "stencil only");
		capture(output, 1);
		outputs.push_back(output);
		draw->Present(PresentMode::FIFO);
		if (iteration == 1 || iteration == 73) {
			std::cout << "MEASURE copy-plane iteration=" << iteration << " requests=" << context.Metrics().copyPlaneRequests
				<< " allocations=" << context.Metrics().transientAllocations[0] << " capacity=" << context.Metrics().copyPlaneCapacity << "\n";
		}
	}
	Check(context.WaitUntilIdle(&error), error);
	for (size_t i = 0; i < outputs.size(); ++i) {
		verify(outputs[i], 0, [i](int) { return std::make_pair(i & 1 ? 0.75f : 0.25f, uint8_t(170)); });
		verify(outputs[i], 1, [i](int) { return std::make_pair(i & 1 ? 0.75f : 0.25f, uint8_t(i + 1)); });
	}
	Check(context.Metrics().copyPlaneRequests == 150, "copy plane request count");
	Check(context.Metrics().transientAllocations[0] == 4 && context.Metrics().copyPlaneCapacity == 8192, "copy plane reuse/shrink");
	// Overlapping copies must snapshot their source before writing the destination.
	for (auto aspect : {Aspect::DEPTH_BIT, Aspect::STENCIL_BIT}) {
		draw->BeginFrame(DebugFlags::NONE);
		clear(src, 0.25f, 77);
		clear(dst, 0.5f, 11);
		draw->CopyFramebufferImage(src, 0, 0, 0, 0, dst, 0, 0, 0, 0, 2, size, 1, aspect, "seed self-copy");
		draw->CopyFramebufferImage(dst, 0, 0, 0, 0, dst, 0, 1, 0, 0, 4, size, 1, aspect, "overlap self-copy");
		id<MTLBuffer> output = [context.Device() newBufferWithLength:4 * planeBytes options:MTLResourceStorageModeShared];
		Check(output != nil, "self copy output allocation");
		capture(output, 0);
		draw->Present(PresentMode::FIFO);
		Check(context.WaitUntilIdle(&error), error);
		verify(output, 0, [aspect](int x) { return std::make_pair(aspect == Aspect::DEPTH_BIT && x < 3 ? 0.25f : 0.5f,
			uint8_t(aspect == Aspect::STENCIL_BIT && x < 3 ? 77 : 11)); });
	}
	for (int i = 0; i < 64; ++i) {
		draw->BeginFrame(DebugFlags::NONE);
		Check(context.BeginCommands(&error), error);
		draw->Present(PresentMode::FIFO);
	}
	Check(context.WaitUntilIdle(&error), error);
	Check(context.Metrics().copyPlaneCapacity == 0, "idle copy plane pool retained");
	std::cout << "PASS copy-plane-reuse 74 async submissions, two layers, both counterparts, two overlapping self-copies, idle retirement\n";
}

class FlatDrawProbe : public DrawEngineMetal {
public:
	using DrawEngineMetal::DrawEngineMetal;
	unsigned flushes = 0;
	void Flush() override { ++flushes; DrawEngineMetal::Flush(); }
	void Hardware(bool enabled) { useHWTransform_ = enabled; }
	void Predecode() { DecodeVerts(dec_, decoded_); }
};

class FramebufferProbe : public FramebufferManagerMetal {
public:
	using FramebufferManagerMetal::FramebufferManagerMetal;
	using FramebufferManagerMetal::FindDisplayVFB;
};

class FlatGPUProbe : public GPUCommonHWMetal {
public:
	using GPUCommonHWMetal::SetPresentationOutput;
	using GPUCommonHWMetal::PresentsToOtherOutput;
	FlatDrawProbe engine;
	FlatGPUProbe(Draw::DrawContext *draw, int msaaLevel) : GPUCommonHWMetal(nullptr, draw), engine(draw) {
		auto *shaders = new ShaderManagerMetal(draw);
		auto *framebuffer = new FramebufferProbe(draw);
		auto *textures = new TextureCacheMetal(draw, framebuffer->GetDraw2D());
		shaderManager_ = shaders;
		framebufferManager_ = framebuffer;
		textureCache_ = textures;
		drawEngineCommon_ = &engine;
		engine.SetGPUCommon(this);
		engine.SetShaderManager(shaders);
		engine.SetFramebufferManager(framebuffer);
		engine.SetTextureCache(textures);
		engine.Init();
		framebuffer->SetShaderManager(shaders);
		framebuffer->SetTextureCache(textures);
		framebuffer->SetDrawEngine(&engine);
		framebuffer->Init(msaaLevel);
		textures->SetFramebufferManager(framebuffer);
		textures->SetShaderManager(shaders);
		gstate_c.SetUseFlags(CheckGPUFeatures() | GPU_ROUND_FRAGMENT_DEPTH_TO_16BIT | GPU_USE_FRAMEBUFFER_ARRAYS);
		textures->NotifyConfigChanged();
	}
	~FlatGPUProbe() override { framebufferManager_->SetDrawEngine(nullptr); }
	void GetStats(StringWriter &) override {}
	FramebufferManagerMetal *Framebuffer() { return framebufferManager_; }
	void ClearResizeNotifications() { displayResized_ = renderResized_ = configChanged_ = false; }
	void Primitive(GEPrimitiveType prim, int count) {
		DisplayList list{};
		list.pc = 0x08800000;
		currentList = &list;
		useFastRunLoop_ = false;
		Execute_Prim((GE_CMD_PRIM << 24) | (prim << 16) | count, 0);
		currentList = nullptr;
	}
};

class PresentationProbe : public PresentationMetal {
public:
	using PresentationMetal::PresentationMetal;
	std::pair<int, int> FirstPostSize() const {
		return postShaderFramebuffers_.empty() ? std::pair{0, 0} :
			std::pair{postShaderFramebuffers_[0]->Width(), postShaderFramebuffers_[0]->Height()};
	}
};

static void OutputPresentation(Draw::DrawContext *draw) {
	using namespace Draw;
	const auto oldConfig = g_Config;
	const auto oldState = gstate;
	const auto oldCached = gstate_c;
	const auto oldCore = PSP_CoreParameter();
	GPUCommon *oldGPU = gpu;
	struct Restore {
		std::function<void()> apply;
		~Restore() { apply(); }
	} restore{[&] { g_Config = oldConfig; gstate = oldState; gstate_c = oldCached; PSP_CoreParameter() = oldCore; gpu = oldGPU; ReloadAllOverlayInfo(); }};
	g_Config.memStickDirectory = Path(std::filesystem::current_path().string() + "/overlay-memstick");
	g_Config.bSkipBufferEffects = false;
	g_Config.bMetalFXSpatial = false;
	g_Config.bStereoRendering = false;
	g_Config.iMultiSampleLevel = 0;
	g_Config.vPostShaderNames.clear();
	GPUStateCache::Reset();
	GEState::Reset();
	const Path folder = GetSysDirectory(DIRECTORY_CUSTOM_OVERLAYS);
	std::filesystem::create_directories(folder.ToString() + "/regression");
	std::vector<uint32_t> image(128 * 128, 0xFF0000FF);
	for (int y = 32; y < 96; ++y) {
		std::fill_n(image.begin() + y * 128 + 32, 64, 0);
	}
	Check(pngSave(folder / "regression.png", (const uint8_t *)image.data(), 128, 128, 4), "overlay fixture PNG");
	Check(pngSave(folder / "regression/background.png", (const uint8_t *)image.data(), 128, 128, 4), "background overlay fixture PNG");
	std::ofstream(folder.ToString() + "/regression/overlay.ini") << "[background]\nImage=background.png\nLayer=Background\nHole=32,32,64,64\n";
	ReloadAllOverlayInfo();
	std::array<uint32_t, 4> green{0xFF00FF00, 0xFF00FF00, 0xFF00FF00, 0xFF00FF00};
	TextureDesc desc{};
	desc.type = TextureType::LINEAR2D;
	desc.format = DataFormat::R8G8B8A8_UNORM;
	desc.width = desc.height = 2;
	desc.depth = desc.mipLevels = 1;
	desc.initData = {reinterpret_cast<const uint8_t *>(green.data())};
	desc.tag = "output presentation source";
	AutoRef<Texture> source(draw->CreateTexture(desc));
	Check(source != nullptr, "presentation source");
	SwapchainDesc outputDesc{WINDOWSYSTEM_METAL_EXT, nullptr, nullptr, 128, 128, "presentation regression"};
	std::string error;
	Swapchain *output = draw->CreateSwapchain(outputDesc, &error);
	Check(output != nullptr, "presentation output");
	struct Destroy {
		DrawContext *draw;
		Swapchain *output;
		~Destroy() { draw->DestroySwapchain(output); }
	} destroy{draw, output};
	PresentationProbe presentation(draw);
	presentation.SetLanguage(GLSL_VULKAN);
	presentation.UpdateDisplaySize(64, 32);
	presentation.UpdateRenderSize(2, 2);
	OutputParams params{};
	params.pixelWidth = params.pixelHeight = 128;
	params.orientation = DeviceOrientation::Landscape;
	presentation.SetOutputOverride(&params);
	DisplayLayoutConfig config{};
	config.fDisplayScale = config.fDisplayAspectRatio = 1.0f;
	config.fDisplayOffsetX = config.fDisplayOffsetY = 0.5f;
	config.bIgnoreScreenInsets = true;
	auto render = [&](bool blank, bool overlay) {
		draw->BeginFrame(DebugFlags::NONE);
		draw->BindFramebufferAsRenderTarget(output, {RPAction::CLEAR, RPAction::CLEAR, RPAction::CLEAR, 0xFF000000, 1.0f, 0}, "output presentation");
		presentation.BeginFrame(config);
		if (!g_Config.vPostShaderNames.empty()) {
			Check(presentation.HasPostShader(), "post-shader was not restored at BeginFrame after device loss");
		}
		if (blank) {
			presentation.SourceBlank();
		} else {
			presentation.SourceTexture(source, 2, 2);
		}
		presentation.RunPostshaderPasses(config, OutputFlags::NEAREST, ROTATION_LOCKED_HORIZONTAL, 0, 0, 1, 1);
		// The UI binds the selected output after the offscreen post-shader passes.
		draw->BindFramebufferAsRenderTarget(output, {RPAction::KEEP, RPAction::KEEP, RPAction::KEEP}, "presentation output rebind");
		presentation.CopyToOutput(config);
		std::vector<uint32_t> pixels(params.pixelWidth * params.pixelHeight);
		Check(draw->CopyFramebufferToMemory(output, Aspect::COLOR_BIT, 0, 0, params.pixelWidth, params.pixelHeight,
			DataFormat::R8G8B8A8_UNORM, pixels.data(), params.pixelWidth, ReadbackMode::BLOCK, "overlay output"), "overlay readback");
		Check((pixels[(params.pixelHeight / 2) * params.pixelWidth + params.pixelWidth / 2] & 0xFFFFFF) == (blank ? 0 : 0x00FF00),
			"game center missing on overridden presentation output");
		if (overlay) {
			Check((pixels[8 * params.pixelWidth + 8] & 0xFFFFFF) == 0x0000FF, "overlay border missing on secondary output");
		}
		draw->Present(PresentMode::FIFO);
	};
	for (const char *overlay : {"regression.png", "regression/background"}) {
		config.sOverlay = overlay;
		render(false, true);
		render(true, true);
	}
	presentation.DeviceLost();
	presentation.DeviceRestore(draw);
	render(false, true);
	config.sOverlay.clear();
	g_Config.vPostShaderNames = {"VideoSmoothingAA", "ColorCorrection"};
	Check(presentation.UpdatePostShader(config), "secondary post-shader setup");
	auto firstSize = presentation.FirstPostSize();
	Check(firstSize.first > 0 && firstSize.second > 0, "secondary post-shader intermediate missing");
	render(false, false);
	presentation.DeviceLost();
	presentation.DeviceRestore(draw);
	Check(!presentation.HasPostShader(), "post-shader restore should be deferred until BeginFrame");
	render(false, false);
	Check(presentation.FirstPostSize() == firstSize, "post-shader intermediate size lost after device restore");
	draw->ResizeSwapchain(output, 256, 256);
	params.pixelWidth = params.pixelHeight = 256;
	presentation.SetOutputOverride(&params);
	render(false, false);
	auto resized = presentation.FirstPostSize();
	Check(resized.first > firstSize.first && resized.second > firstSize.second, "output resize did not rebuild post-shader intermediates");
	FlatGPUProbe testGPU(draw, 0);
	gpu = &testGPU;
	testGPU.SetPresentationOutput(&params);
	Check(testGPU.PresentsToOtherOutput(), "Metal GE output override did not reach its private presentation");
	testGPU.SetPresentationOutput(nullptr);
	Check(!testGPU.PresentsToOtherOutput(), "Metal GE output override was not cleared");
	std::cout << "PASS output presentation: foreground/background/blank overlays, device restore, two post passes rebuilt at output resize, private GE routing\n";
}

static void UpstreamVideoClamp(Draw::DrawContext *draw) {
	const auto oldState = gstate;
	const auto oldCached = gstate_c;
	GPUCommon *oldGPU = gpu;
	struct Restore {
		std::function<void()> apply;
		~Restore() { apply(); }
	} restore{[&] { gstate = oldState; gstate_c = oldCached; gpu = oldGPU; }};
	CommandProbe testGPU(draw);
	gpu = &testGPU;
	GEState::Reset();
	GPUStateCache::Reset();
	int cases = 0;
	for (GEBufferFormat format : {GE_FORMAT_8888, GE_FORMAT_565}) {
		const int bpp = BufferFormatBytesPerPixel(format);
		for (auto dimensions : {std::pair{256, 480}, std::pair{480, 272}, std::pair{128, 64}, std::pair{512, 512}}) {
			MetalTexCacheEntry entry{};
			entry.addr = 0x08840000;
			entry.bufw = 512;
			entry.format = (GETextureFormat)format;
			entry.status = MetalTexStatus::VIDEO;
			gstate.texsize[0] = 9 | (9 << 8);
			gstate_c.curTextureWidth = gstate_c.curTextureHeight = 512;
			testGPU.PerformWriteFormattedFromMemory(entry.addr, 512 * bpp * dimensions.second, 512, format, dimensions.first);
			gstate_c.Clean(DIRTY_TEXCLAMP | DIRTY_FRAGMENTSHADER_STATE);
			testGPU.Cache().ApplyTextureFinish(&entry, false);
			Check(gstate_c.videoColumns == dimensions.first && gstate_c.videoRows == dimensions.second,
				"Metal video footprint lost written rows/columns");
			Check(gstate_c.IsDirty(DIRTY_TEXCLAMP) && gstate_c.IsDirty(DIRTY_FRAGMENTSHADER_STATE),
				"video size change did not dirty uniforms and shader selection");
			float clamp[4], offset[2];
			CalcTexClamp(clamp, offset);
			Check(clamp[0] == dimensions.first / 512.0f && clamp[1] == dimensions.second / 512.0f,
				"video clamp dimensions");
			Check(VideoClampsS() == (dimensions.first < 512) && VideoClampsT() == (dimensions.second < 512),
				"full video dimension must retain wrapping");
			gstate_c.Clean(DIRTY_TEXCLAMP | DIRTY_FRAGMENTSHADER_STATE);
			testGPU.Cache().ApplyTextureFinish(&entry, false);
			Check(!gstate_c.IsDirty(DIRTY_TEXCLAMP | DIRTY_FRAGMENTSHADER_STATE), "unchanged video size dirtied shader");
			++cases;
		}
	}
	MetalTexCacheEntry unknown{};
	unknown.addr = 0x09F00000;
	unknown.bufw = 512;
	unknown.format = GE_TFMT_8888;
	unknown.status = MetalTexStatus::VIDEO;
	testGPU.Cache().ApplyTextureFinish(&unknown, false);
	Check(VideoFrameWidth() == 480 && VideoFrameHeight() == 272, "unknown video footprint fallback");
	std::cout << "PASS upstream video footprint: " << cases << " size/format changes, stable cache and unknown-size fallback\n";
}

static void UpstreamFramebuffers(Draw::DrawContext *draw) {
	const auto oldState = gstate;
	const auto oldCached = gstate_c;
	const auto oldConfig = g_Config;
	const auto oldCore = PSP_CoreParameter();
	GPUCommon *oldGPU = gpu;
	struct Restore {
		std::function<void()> apply;
		~Restore() { apply(); }
	} restore{[&] { gstate = oldState; gstate_c = oldCached; g_Config = oldConfig; PSP_CoreParameter() = oldCore; gpu = oldGPU; }};
	GEState::Reset();
	GPUStateCache::Reset();
	g_Config.bSkipBufferEffects = false;
	g_Config.bStereoRendering = false;
	g_Config.iInternalResolution = 8;
	g_Config.iMultiSampleLevel = 0;
	g_Config.iBloomHack = 0;
	auto &core = PSP_CoreParameter();
	core.renderWidth = 480 * 8;
	core.renderHeight = 272 * 8;
	core.renderScaleFactor = 8;
	auto &compat = const_cast<CompatFlags &>(core.compat.flags());
	compat = {};
	compat.FixedFramebuffer = {0x04060000, 128, 96};
	compat.NativeResolutionFramebuffer = 0x04060000;
	FlatGPUProbe testGPU(draw, 0);
	gpu = &testGPU;
	auto *framebuffer = static_cast<FramebufferProbe *>(testGPU.Framebuffer());
	FramebufferHeuristicParams params{};
	params.fb_address = 0x04060000;
	params.z_address = 0x040E0000;
	params.fb_stride = params.z_stride = 128;
	params.fb_format = GE_FORMAT_8888;
	params.viewportWidth = params.regionWidth = params.scissorRight = 32;
	params.viewportHeight = params.regionHeight = params.scissorBottom = 32;
	// The first operation can be a partial clear; untouched memory must still be uploaded.
	auto *memory = reinterpret_cast<uint32_t *>(Memory::GetPointerWriteUnchecked(params.fb_address));
	for (int i = 0; i < 128 * 96; ++i) {
		memory[i] = 0xFF000000 | (i * 17 & 0xFFFFFF);
	}
	auto *fixed = framebuffer->DoSetRenderFrameBuffer(params, 0);
	Check(fixed && fixed->fbo && fixed->bufferWidth == 128 && fixed->bufferHeight == 96, "fixed framebuffer initial extent");
	Check(fixed->renderWidth == 128 && fixed->renderHeight == 96 && fixed->renderScaleFactor == 1,
		"native-resolution framebuffer scaled at 8x");
	std::vector<uint32_t> pixels(128 * 96);
	Check(draw->CopyFramebufferToMemory(fixed->fbo, Draw::Aspect::COLOR_BIT, 0, 0, 128, 96,
		Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), 128, Draw::ReadbackMode::BLOCK, "fixed footprint"), "fixed framebuffer readback");
	Check(std::equal(pixels.begin(), pixels.end(), memory), "fixed framebuffer lost memory outside initial draw");
	framebuffer->SetDisplayFramebuffer(params.fb_address, 128, GE_FORMAT_8888);
	fixed->height = 51;
	u32 offsetX, offsetY;
	Check(framebuffer->FindDisplayVFB(params.fb_address, &offsetX, &offsetY) == fixed, "small final draw rejected a full-size display buffer");
	fixed->bufferHeight = 63;
	Check(framebuffer->FindDisplayVFB(params.fb_address, &offsetX, &offsetY) == nullptr, "undersized display buffer accepted");
	fixed->bufferHeight = 96;
	params.fb_address = 0x04100000;
	params.isDrawing = true;
	auto *normal = framebuffer->DoSetRenderFrameBuffer(params, 0);
	Check(normal && normal->fbo && normal->renderScaleFactor == 8, "unrelated framebuffer lost 8x scaling");
	std::cout << "PASS upstream framebuffer extents: fixed partial-clear upload, native 1x at global 8x, unaffected 8x target, display selection\n";
}

static void UpstreamLogicDraws(Draw::DrawContext *draw) {
	const auto oldState = gstate;
	const auto oldCached = gstate_c;
	const auto oldConfig = g_Config;
	const auto oldCore = PSP_CoreParameter();
	GPUCommon *oldGPU = gpu;
	struct Restore {
		std::function<void()> apply;
		~Restore() { apply(); }
	} restore{[&] { gstate = oldState; gstate_c = oldCached; g_Config = oldConfig; PSP_CoreParameter() = oldCore; gpu = oldGPU; }};
	GEState::Reset();
	GPUStateCache::Reset();
	g_Config.bHardwareTransform = true;
	g_Config.bSkipBufferEffects = false;
	g_Config.bStereoRendering = false;
	g_Config.iInternalResolution = 1;
	g_Config.iMultiSampleLevel = 0;
	g_Config.iDepthRasterMode = (int)DepthRasterMode::OFF;
	auto &core = PSP_CoreParameter();
	core.renderWidth = 480;
	core.renderHeight = 272;
	core.renderScaleFactor = 1;
	auto &compat = const_cast<CompatFlags &>(core.compat.flags());
	compat = {};
	compat.SeparateLogicOpDraws = true;
	FlatGPUProbe testGPU(draw, 0);
	gpu = &testGPU;
	// Exercise the snapshot fallback used when coherent framebuffer fetch is unavailable.
	gstate_c.SetUseFlags(gstate_c.GetUseFlags() & ~(GPU_USE_FRAMEBUFFER_FETCH | GPU_USE_LOGIC_OP));
	gstate.region2 = gstate.scissor2 = 63 | (63 << 10);
	gstate.fbptr = 0x40000;
	gstate.fbwidth = gstate.zbwidth = 64;
	gstate.zbptr = 0x80000;
	gstate.framebufpixformat = GE_FORMAT_8888;
	gstate.maxz = 65535;
	gstate.materialalpha = 255;
	gstate.shademodel = GE_SHADE_GOURAUD;
	gstate.vertType = GE_VTYPE_COL_8888 | GE_VTYPE_POS_FLOAT | GE_VTYPE_THROUGH;
	gstate.logicOpEnable = 1;
	gstate.lop = GE_LOGIC_XOR;
	bool changed;
	auto *vfb = testGPU.Framebuffer()->SetRenderFrameBuffer(true, 0, &changed);
	Check(vfb && vfb->fbo, "logic fallback framebuffer");
	draw->BeginFrame(Draw::DebugFlags::NONE);
	draw->BindFramebufferAsRenderTarget(vfb->fbo, {Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR,
		0, 1.0f, 0}, "logic fallback clear");
	struct Vertex { uint32_t color; float x, y, z; };
	const uint32_t color = 0xFF2468AC;
	const std::array<Vertex, 6> vertices{{{color, 8, 8, 1}, {color, 32, 32, 1},
		{color, 16, 16, 1}, {color, 40, 40, 1}, {color, 8, 8, 1}, {color, 32, 32, 1}}};
	gstate_c.vertexAddr = 0x08880000;
	memcpy(Memory::GetPointerWriteUnchecked(gstate_c.vertexAddr), vertices.data(), sizeof(vertices));
	const auto flushes = testGPU.engine.flushes;
	testGPU.Primitive(GE_PRIM_RECTANGLES, 6);
	Check(testGPU.engine.flushes >= flushes + 3, "overlapping snapshot rectangles were batched together");
	Check(gstate_c.vertexAddr == 0x08880000 + sizeof(vertices), "separate draws changed vertex-address advancement");
	std::array<uint32_t, 64 * 64> pixels{};
	Check(draw->CopyFramebufferToMemory(vfb->fbo, Draw::Aspect::COLOR_BIT, 0, 0, 64, 64,
		Draw::DataFormat::R8G8B8A8_UNORM, pixels.data(), 64, Draw::ReadbackMode::BLOCK, "logic fallback result"), "logic fallback readback");
	for (int y = 0; y < 64; ++y) {
		for (int x = 0; x < 64; ++x) {
			const uint32_t expected = x >= 16 && x < 40 && y >= 16 && y < 40 ? color & 0xFFFFFF : 0;
			Check((pixels[y * 64 + x] & 0xFFFFFF) == expected, "ordered XOR rectangles differ from CPU oracle at " + std::to_string(x) + "," + std::to_string(y));
		}
	}
	draw->Present(Draw::PresentMode::FIFO);
	std::cout << "PASS upstream logic fallback: three overlapping XOR rectangles, 4096 CPU-reference pixels and vertex advancement\n";
}

static void BufferEffectsChanges(Draw::DrawContext *draw) {
	const auto oldState = gstate;
	const auto oldCached = gstate_c;
	const auto oldConfig = g_Config;
	const auto oldCore = PSP_CoreParameter();
	GPUCommon *oldGPU = gpu;
	struct Restore {
		std::function<void()> apply;
		~Restore() { apply(); }
	} restore{[&] {
		gstate = oldState;
		gstate_c = oldCached;
		g_Config = oldConfig;
		PSP_CoreParameter() = oldCore;
		gpu = oldGPU;
	}};
	g_Config.bSkipBufferEffects = false;
	g_Config.bStereoRendering = false;
	g_Config.iInternalResolution = 1;
	g_Config.iMultiSampleLevel = 0;
	PSP_CoreParameter().renderWidth = 480;
	PSP_CoreParameter().renderHeight = 272;
	PSP_CoreParameter().renderScaleFactor = 1;
	GPUStateCache::Reset();
	GEState::Reset();
	FlatGPUProbe testGPU(draw, 0);
	gpu = &testGPU;
	DisplayLayoutConfig config{};
	for (bool skip : {true, true, false, false, true, false}) {
		testGPU.ClearResizeNotifications();
		g_Config.bSkipBufferEffects = skip;
		testGPU.BeginHostFrame(config);
		Check(testGPU.Framebuffer()->UseBufferedRendering() == !skip,
			"buffer effects change ignored without resize notification");
	}
	std::cout << "PASS buffer-effects changes: six frames, both directions, no resize notification\n";
}

static void NearClipCapacity() {
	const auto oldState = gstate;
	const auto oldCached = gstate_c;
	const auto oldConfig = g_Config;
	struct Restore {
		std::function<void()> apply;
		~Restore() { apply(); }
	} restore{[&] { gstate = oldState; gstate_c = oldCached; g_Config = oldConfig; }};
	GPUStateCache::Reset();
	GEState::Reset();
	g_Config.bSmart2DTexFiltering = false;
	g_Config.iTexScalingLevel = 1;
	gstate_c.curTextureWidth = gstate_c.curTextureHeight = 1;
	for (int i : {0, 4, 8}) {
		gstate.worldMatrix[i] = gstate.viewMatrix[i] = 1.0f;
	}
	for (int i : {0, 5, 10, 15}) {
		gstate.projMatrix[i] = 1.0f;
	}
	auto f24 = [](float value) { uint32_t bits; memcpy(&bits, &value, sizeof(bits)); return bits >> 8; };
	gstate.viewportxscale = f24(32);
	gstate.viewportyscale = f24(-32);
	gstate.viewportzscale = f24(32767);
	gstate.viewportxcenter = gstate.viewportycenter = f24(32);
	gstate.viewportzcenter = f24(32768);
	gstate.depthClipEnable = 1;
	gstate.maxz = 65535;
	gstate.materialalpha = 255;
	DecVtxFormat format{};
	format.stride = 12;
	int cases = 0;
	for (int outside = 0; outside <= 3; ++outside) {
		float vertices[3][3] = {{-0.5f, -0.5f, 0}, {0.5f, -0.5f, 0}, {0, 0.5f, 0}};
		for (int i = 0; i < outside; ++i) {
			vertices[i][2] = -2.0f;
		}
		for (int vertexCapacity = 3; vertexCapacity <= 6; ++vertexCapacity) {
			for (int indexCapacity = 0; indexCapacity <= 6; ++indexCapacity) {
				std::vector<TransformedVertex> transformed(vertexCapacity + 4);
				memset(transformed.data(), 0xA5, transformed.size() * sizeof(TransformedVertex));
				std::vector<uint16_t> indices(3 + indexCapacity + 8, 0xA55A);
				indices[0] = 0; indices[1] = 1; indices[2] = 2;
				uint16_t *drawIndices = indices.data();
				SoftwareTransformParams params{};
				params.decoded = (uint8_t *)vertices;
				params.transformed = params.transformedExpanded = transformed.data();
				params.clipInfoFlags = ClipInfoFlags::Valid | ClipInfoFlags::SoftClipCull;
				SoftwareTransformResult result{};
				RunSoftwareTransform(params, GE_PRIM_TRIANGLES, GE_VTYPE_POS_FLOAT, format,
					3, vertexCapacity, 3, drawIndices, 3 + indexCapacity, &result);
				const auto *guard = (const uint8_t *)(transformed.data() + vertexCapacity);
				Check(std::all_of(guard, guard + 4 * sizeof(TransformedVertex), [](uint8_t byte) { return byte == 0xA5; }),
					"near clipping wrote past vertex capacity");
				Check(std::all_of(indices.begin() + 3 + indexCapacity, indices.end(), [](uint16_t index) { return index == 0xA55A; }),
					"near clipping wrote past index capacity");
				const int expected = outside == 0 ? (indexCapacity >= 3 ? 3 : 0) :
					outside == 3 ? 0 : (vertexCapacity >= 5 && indexCapacity >= 6 ? (outside == 1 ? 6 : 3) : 0);
				Check(result.drawIndexCount == expected && result.drawVertexCount <= vertexCapacity,
					"near clipping capacity changed accepted topology");
				for (int i = 0; i < result.drawIndexCount; ++i) {
					Check(drawIndices[i] < result.drawVertexCount, "near clipping produced an invalid index");
				}
				++cases;
			}
		}
	}
	std::cout << "PASS near-clip capacities: " << cases << " guarded vertex/index limits and accepted topology\n";
}

static void FlatShading(Draw::DrawContext *draw, Metal::RenderManager *manager, int msaaLevel) {
	Check([manager->Context().Device() supportsTextureSampleCount:1 << msaaLevel], "flat test sample count");
	std::cout << "BEGIN flat-shading samples=" << (1 << msaaLevel) << "\n";
	const auto oldState = gstate;
	const auto oldCached = gstate_c;
	const auto oldConfig = g_Config;
	GPUCommon *oldGPU = gpu;
	auto &core = PSP_CoreParameter();
	const int oldWidth = core.renderWidth, oldHeight = core.renderHeight, oldScale = core.renderScaleFactor;
	struct Cleanup {
		std::function<void()> restore;
		~Cleanup() { restore(); }
	} cleanup{[&] {
		gpu = oldGPU;
		gstate = oldState;
		gstate_c = oldCached;
		g_Config = oldConfig;
		core.renderWidth = oldWidth;
		core.renderHeight = oldHeight;
		core.renderScaleFactor = oldScale;
	}};
	g_Config.bHardwareTransform = true;
	g_Config.bSkipBufferEffects = false;
	g_Config.bShaderCache = false;
	g_Config.bStereoRendering = false;
	g_Config.iInternalResolution = 1;
	g_Config.iMultiSampleLevel = msaaLevel;
	g_Config.iDepthRasterMode = (int)DepthRasterMode::OFF;
	core.renderWidth = 480;
	core.renderHeight = 272;
	core.renderScaleFactor = 1;
	GPUStateCache::Reset();
	GEState::Reset();
	FlatGPUProbe testGPU(draw, msaaLevel);
	gpu = &testGPU;
	for (int i : {0, 4, 8}) {
		gstate.worldMatrix[i] = gstate.viewMatrix[i] = 1.0f;
	}
	for (int i : {0, 5, 10, 15}) {
		gstate.projMatrix[i] = 1.0f;
	}
	auto f24 = [](float f) { uint32_t bits; memcpy(&bits, &f, sizeof(bits)); return bits >> 8; };
	gstate.viewportxscale = f24(32);
	gstate.viewportyscale = f24(-32);
	gstate.viewportzscale = f24(32767);
	gstate.viewportxcenter = gstate.viewportycenter = f24(32);
	gstate.viewportzcenter = f24(32768);
	gstate.region2 = gstate.scissor2 = 63 | (63 << 10);
	gstate.fbwidth = gstate.zbwidth = 64;
	gstate.zbptr = 0x10000;
	gstate.framebufpixformat = GE_FORMAT_8888;
	gstate.maxz = 65535;
	gstate.zTestEnable = 1;
	gstate.ztestfunc = GE_COMP_ALWAYS;
	gstate.materialalpha = 255;
	gstate.shademodel = GE_SHADE_FLAT;
	gstate.vertType = GE_VTYPE_COL_8888 | GE_VTYPE_POS_FLOAT;
	bool changed;
	auto *vfb = testGPU.Framebuffer()->SetRenderFrameBuffer(true, 0, &changed);
	Check(vfb && vfb->fbo, "flat test framebuffer");
	struct Vertex { uint32_t color; float x, y, z; };
	std::array<Vertex, 3> vertices{{{0xFF0000FF, -0.75f, -0.75f, -0.25f},
		{0xFF00FF00, 0.75f, -0.75f, 0.25f}, {0xFFFF0000, 0.0f, 0.75f, 0.0f}}};
	std::array<std::array<uint32_t, 64 * 64>, 2> colors{};
	std::array<std::array<float, 64 * 64>, 2> depths{};
	for (int hw = 0; hw < 2; ++hw) {
		testGPU.engine.Hardware(hw);
		manager->EndRenderPass();
		std::string error;
		if (!manager->Context().Commands()) {
			Check(manager->Context().BeginCommands(&error), error);
		}
		draw->BindFramebufferAsRenderTarget(vfb->fbo, {Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR,
			0, 1.0f, 0, "flat test"}, "flat test");
		gstate_c.Dirty(DIRTY_ALL_RENDER_STATE | DIRTY_ALL_UNIFORMS);
		int bytesRead = 0;
		testGPU.engine.DispatchSubmitPrim(vertices.data(), nullptr, GE_PRIM_TRIANGLES, vertices.size(), gstate.vertType,
			true, &bytesRead, ClipInfoFlags::Valid);
		testGPU.engine.Flush();
		Check(testGPU.engine.LastError().empty(), testGPU.engine.LastError());
		Check(draw->CopyFramebufferToMemory(vfb->fbo, Draw::Aspect::COLOR_BIT, 0, 0, 64, 64,
			Draw::DataFormat::R8G8B8A8_UNORM, colors[hw].data(), 64, Draw::ReadbackMode::BLOCK, "flat color"), "flat color readback");
		Check(draw->CopyFramebufferToMemory(vfb->fbo, Draw::Aspect::DEPTH_BIT, 0, 0, 64, 64,
			Draw::DataFormat::D32F, depths[hw].data(), 64, Draw::ReadbackMode::BLOCK, "flat depth"), "flat depth readback");
	}
	Check(colors[0] == colors[1] && depths[0] == depths[1], "flat HW/SW pixels or depth differ");
	const int drawn = std::count_if(colors[0].begin(), colors[0].end(), [](uint32_t pixel) { return (pixel & 0xFFFFFF) == 0xFF0000; });
	Check(drawn > 500, "flat test did not draw the provoking vertex color");
	Check(std::count_if(depths[0].begin(), depths[0].end(), [](float d) { return d < 1.0f && d > 0.0f; }) >= drawn,
		"flat test did not write varying depth");
	Check(testGPU.engine.TransformMetrics().flatDraws == 2 && testGPU.engine.TransformMetrics().flatHardwareEligible == 1,
		"flat test did not reach both transform policies");
	Check(testGPU.engine.TransformMetrics().flatHardwareDraws == 1, "flat test did not reach hardware transform");
	std::cout << "PASS flat-shading initial HW/SW candidate; " << drawn << " blue pixels, identical depth\n";
	auto render = [&](const auto &mesh, GEPrimitiveType primitive, const void *indexData,
		bool clockwise, bool predecoded, ClipInfoFlags flags, int hw, bool afterReadback = false) {
		testGPU.engine.Hardware(hw);
		manager->EndRenderPass();
		std::string error;
		if (!manager->Context().Commands()) {
			Check(manager->Context().BeginCommands(&error), error);
		}
		draw->BindFramebufferAsRenderTarget(vfb->fbo, {Draw::RPAction::CLEAR, Draw::RPAction::CLEAR, Draw::RPAction::CLEAR,
			0, 1.0f, 0, "flat variants"}, "flat variants");
		if (afterReadback) {
			std::array<uint32_t, 64 * 64> cleared{};
			Check(draw->CopyFramebufferToMemory(vfb->fbo, Draw::Aspect::COLOR_BIT, 0, 0, 64, 64,
				Draw::DataFormat::R8G8B8A8_UNORM, cleared.data(), 64, Draw::ReadbackMode::BLOCK,
				"before direct vertex upload"), "pre-draw readback");
			Check(!manager->Context().Commands(), "readback did not submit the command buffer");
		}
		gstate_c.Dirty(DIRTY_ALL_RENDER_STATE | DIRTY_ALL_UNIFORMS);
		int bytesRead = 0;
		testGPU.engine.DispatchSubmitPrim(mesh.data(), indexData, primitive, mesh.size(), gstate.vertType,
			clockwise, &bytesRead, flags);
		if (predecoded) {
			testGPU.engine.Predecode();
		}
		testGPU.engine.Flush();
		Check(testGPU.engine.LastError().empty(), testGPU.engine.LastError());
		Check(draw->CopyFramebufferToMemory(vfb->fbo, Draw::Aspect::COLOR_BIT, 0, 0, 64, 64,
			Draw::DataFormat::R8G8B8A8_UNORM, colors[hw].data(), 64, Draw::ReadbackMode::BLOCK, "flat color"), "flat color readback");
		Check(draw->CopyFramebufferToMemory(vfb->fbo, Draw::Aspect::DEPTH_BIT, 0, 0, 64, 64,
			Draw::DataFormat::D32F, depths[hw].data(), 64, Draw::ReadbackMode::BLOCK, "flat depth"), "flat depth readback");
	};
	for (bool predecoded : {false, true}) {
		render(vertices, GE_PRIM_TRIANGLES, nullptr, true, predecoded, ClipInfoFlags::Valid, 0);
		render(vertices, GE_PRIM_TRIANGLES, nullptr, true, predecoded, ClipInfoFlags::Valid, 1, true);
		Check(colors[0] == colors[1] && depths[0] == depths[1], "vertex upload after readback lost color/depth");
	}
	std::cout << "PASS vertex upload after blocking readback: direct and predecoded HW/SW color/depth pairs\n";
	int cases = 0;
	for (GEPrimitiveType primitive : {GE_PRIM_TRIANGLES, GE_PRIM_TRIANGLE_STRIP, GE_PRIM_TRIANGLE_FAN}) {
		std::vector<Vertex> mesh;
		if (primitive == GE_PRIM_TRIANGLES) {
			mesh.assign(vertices.begin(), vertices.end());
		} else if (primitive == GE_PRIM_TRIANGLE_STRIP) {
			mesh = {{0xFF0000FF, -0.75f, -0.75f, -0.25f}, {0xFF00FF00, 0.75f, -0.75f, 0.25f},
				{0xFFFF0000, -0.75f, 0.75f, -0.25f}, {0xFF00FFFF, 0.75f, 0.75f, 0.25f}};
		} else {
			mesh = {{0xFF0000FF, 0, 0, 0}, {0xFF00FF00, -0.75f, -0.75f, -0.25f},
				{0xFFFF0000, 0.75f, -0.75f, 0.25f}, {0xFF00FFFF, 0.75f, 0.75f, 0.25f},
				{0xFFFF00FF, -0.75f, 0.75f, -0.25f}};
		}
		std::vector<uint8_t> inds8;
		std::vector<uint16_t> inds16;
		std::vector<uint32_t> inds32;
		for (int i = 0; i < (int)mesh.size(); ++i) {
			inds8.push_back(i);
			inds16.push_back(i);
			inds32.push_back(i);
		}
		const std::array<const void *, 4> indexData{nullptr, inds8.data(), inds16.data(), inds32.data()};
		for (int indexType = 0; indexType < 4; ++indexType) {
			for (bool clockwise : {false, true}) {
				for (int cull = 0; cull < 3; ++cull) {
					for (bool predecoded : {false, true}) {
						const std::string label = "prim=" + std::to_string(primitive) + " index=" + std::to_string(indexType) +
							" clockwise=" + std::to_string(clockwise) + " cull=" + std::to_string(cull) + " predecoded=" + std::to_string(predecoded);
						gstate.vertType = GE_VTYPE_COL_8888 | GE_VTYPE_POS_FLOAT | (indexType << GE_VTYPE_IDX_SHIFT);
						gstate.cullfaceEnable = cull != 0;
						gstate.cullmode = std::max(0, cull - 1);
						const auto startMetrics = testGPU.engine.TransformMetrics();
						for (int hw = 0; hw < 2; ++hw) {
							render(mesh, primitive, indexData[indexType], clockwise, predecoded, ClipInfoFlags::Valid, hw);
						}
						Check(colors[0] == colors[1], label + " HW/SW color differs");
						Check(depths[0] == depths[1], label + " HW/SW depth differs");
						if (cull == 0) {
							Check(std::any_of(colors[0].begin(), colors[0].end(), [](uint32_t c) { return (c & 0xFFFFFF) != 0; }), label + " no output");
						}
						Check(testGPU.engine.TransformMetrics().flatHardwareDraws == startMetrics.flatHardwareDraws + 1,
							label + " missing hardware draw");
						++cases;
					}
				}
			}
		}
	}
	std::cout << "PASS flat-shading topology/index/cull/upload " << cases << " HW/SW pixel/depth pairs\n";
	gstate.vertType = GE_VTYPE_COL_8888 | GE_VTYPE_POS_FLOAT;
	gstate.cullfaceEnable = 0;
	for (int mode = 0; mode < 5; ++mode) {
		std::vector<Vertex> mesh(vertices.begin(), vertices.end());
		ClipInfoFlags flags = ClipInfoFlags::Valid;
		gstate.minz = 0;
		gstate.maxz = 65535;
		gstate.lightingEnable = mode == 4;
		if (mode == 0) {
			mesh[0].x = -1.75f;
		} else if (mode == 1) {
			mesh[0].z = -1.25f;
			flags |= ClipInfoFlags::SoftClipCull;
		} else if (mode == 2) {
			for (auto &vertex : mesh) {
				vertex.z = 2.0f;
			}
			flags |= ClipInfoFlags::SoftClipCull;
		} else if (mode == 3) {
			mesh[1].z = 0.8f;
			gstate.minz = 10000;
			gstate.maxz = 50000;
			flags |= ClipInfoFlags::MinMaxZDiscard;
		} else {
			gstate.materialupdate = 1;
			gstate.materialambient = 0x6789AB;
			gstate.materialemissive = 0x132351;
			gstate.ambientcolor = 0x93735B;
			gstate.ambientalpha = 255;
			mesh[2].color = 0xFF8DA9C7;
		}
		const auto before = testGPU.engine.TransformMetrics();
		for (int hw = 0; hw < 2; ++hw) {
			render(mesh, GE_PRIM_TRIANGLES, nullptr, true, false, flags, hw);
		}
		Check(colors[0] == colors[1], "flat clipping/lighting mode=" + std::to_string(mode) + " colors differ");
		Check(depths[0] == depths[1], "flat clipping/lighting mode=" + std::to_string(mode) + " depth differs");
		Check(testGPU.engine.TransformMetrics().flatHardwareDraws == before.flatHardwareDraws + (mode == 1 || mode == 2 || mode == 4 ? 0 : 1),
			"flat clipping/lighting transform policy mismatch");
		std::cout << "PASS flat-shading clipping/lighting mode=" << mode << "\n";
	}
	struct LitVertex { uint32_t color; float normal[3]; float position[3]; };
	std::vector<LitVertex> lit;
	for (const auto &vertex : vertices) {
		lit.push_back({vertex.color, {0.0f, 0.0f, 1.0f}, {vertex.x, vertex.y, vertex.z}});
	}
	gstate.vertType = GE_VTYPE_COL_8888 | GE_VTYPE_NRM_FLOAT | GE_VTYPE_POS_FLOAT;
	gstate.materialupdate = 0;
	gstate.materialambient = 0x605040;
	gstate.materialdiffuse = 0xAA9876;
	gstate.materialspecular = 0x424242;
	gstate.materialspecularcoef = f24(3.0f);
	gstate.materialemissive = 0;
	gstate.ambientcolor = 0x161718;
	gstate.lightEnable[0] = 1;
	gstate.lpos[0] = f24(0.25f);
	gstate.lpos[1] = f24(0.5f);
	gstate.lpos[2] = f24(2.0f);
	gstate.ldir[0] = f24(0.0f);
	gstate.ldir[1] = f24(0.0f);
	gstate.ldir[2] = f24(1.0f);
	gstate.latt[0] = f24(0.5f);
	gstate.latt[1] = f24(0.2f);
	gstate.latt[2] = f24(0.1f);
	gstate.lconv[0] = f24(2.0f);
	gstate.lcutoff[0] = f24(0.25f);
	gstate.lcolor[0] = 0x181A17;
	gstate.lcolor[1] = 0xAE9072;
	gstate.lcolor[2] = 0x7B818B;
	for (int lightType = 0; lightType < 3; ++lightType) {
		gstate.ltype[0] = (lightType << 8) | GE_LIGHTCOMP_BOTH;
		for (int secondary = 0; secondary < 2; ++secondary) {
			gstate.lmode = secondary;
			const auto before = testGPU.engine.TransformMetrics();
			for (int hw = 0; hw < 2; ++hw) {
				render(lit, GE_PRIM_TRIANGLES, nullptr, true, false, ClipInfoFlags::Valid, hw);
			}
			if (colors[0] != colors[1]) {
				const auto mismatch = std::mismatch(colors[0].begin(), colors[0].end(), colors[1].begin());
				std::cout << "MEASURE flat-lit first-mismatch software=" << std::hex << *mismatch.first
					<< " hardware=" << *mismatch.second << std::dec << " depth-equal=" << (depths[0] == depths[1]) << "\n";
			}
			Check(colors[0] == colors[1] && depths[0] == depths[1], "flat light type=" + std::to_string(lightType) +
				" secondary=" + std::to_string(secondary) + " output differs");
			Check(testGPU.engine.TransformMetrics().flatHardwareDraws == before.flatHardwareDraws &&
				testGPU.engine.TransformMetrics().flatLightingFallbacks == before.flatLightingFallbacks + 1,
				"flat-lit precision fallback not reached");
			std::cout << "PASS flat-shading light=" << lightType << " secondary=" << secondary << "\n";
		}
	}
	if (msaaLevel != 0) {
		return;
	}
	gstate.lightingEnable = 0;
	gstate.vertType = GE_VTYPE_COL_8888 | GE_VTYPE_POS_FLOAT;
	std::vector<Vertex> batch;
	batch.reserve(9000);
	for (int i = 0; i < 3000; ++i) {
		batch.insert(batch.end(), vertices.begin(), vertices.end());
	}
	std::array<std::vector<double>, 2> samples;
	for (int sample = 0; sample < 80; ++sample) {
		for (int phase = 0; phase < 2; ++phase) {
			const int hw = (phase + sample) & 1;
			const auto before = testGPU.engine.TransformMetrics();
			render(batch, GE_PRIM_TRIANGLES, nullptr, true, false, ClipInfoFlags::Valid, hw);
			const auto after = testGPU.engine.TransformMetrics();
			Check(after.flatVertices - before.flatVertices == 9000 && after.flatHardwareDraws - before.flatHardwareDraws == hw,
				"flat benchmark did not reach requested transform path");
			if (sample >= 8) {
				samples[hw].push_back((after.flatPreparationSeconds - before.flatPreparationSeconds) * 1000.0);
			}
		}
		Check(colors[0] == colors[1] && depths[0] == depths[1], "flat benchmark output changed");
	}
	std::sort(samples[0].begin(), samples[0].end());
	std::sort(samples[1].begin(), samples[1].end());
	for (double quantile : {0.5, 0.95, 0.99}) {
		const size_t i = (size_t)std::ceil(quantile * samples[0].size()) - 1;
		std::cout << "MEASURE flat-preparation p=" << quantile << " software-ms=" << samples[0][i]
			<< " hardware-ms=" << samples[1][i] << " vertices=9000 samples=" << samples[0].size() << "\n";
	}
	std::cout << "PASS flat-shading 9000-vertex benchmark preserves color/depth in 80 alternating pairs\n";
}

static void ScalingSupport(Draw::DrawContext *draw) {
	int cases = 0;
	for (int scale = 0; scale <= 8; ++scale) {
		for (int passes = 0; passes <= 5; ++passes) {
			for (bool constants : {false, true}) {
				MetalTextureShaderInfo info;
				info.scaleFactor = scale;
				info.constantBuffer = constants ? Path("constants.bin") : Path();
				info.computeShaderFiles.assign(passes, Path("stage.comp"));
				bool expected = scale >= 2 && scale < 8 &&
					(passes == 1 || (scale == 2 && passes == 3) || (scale == 4 && passes == 4));
				Check(MetalTextureShaderSupported(info) == expected, "scaler structural support mismatch");
				if (!info.computeShaderFiles.empty()) {
					info.computeShaderFiles[0] = Path();
					Check(!MetalTextureShaderSupported(info), "empty stage accepted");
				}
				++cases;
			}
		}
	}
	MetalReloadAllPostShaderInfo(draw);
	for (const auto &info : MetalGetAllTextureShaderInfo()) {
		Check(MetalTextureShaderSupported(info.section) == (info.metalSupported && MetalTextureShaderSupported(info)),
			"selection/configure support drift");
	}
	std::cout << "PASS scaler-common-support " << cases << " plus registered assets\n";
}

int main(int argc, char **argv) {
	std::cout << std::unitbuf;
	static bool logEnabled = getenv("METAL_TEST_LOG") != nullptr;
	g_logManager.Init(&logEnabled, true);
	g_logManager.SetAllLogLevels(LogLevel::LWARNING);
	g_logManager.SetOutputsEnabled(LogOutput::Stdio);
	@autoreleasepool {
		try {
			Check(argc == 2, "pass the repository root");
			const auto repository = std::filesystem::canonical(argv[1]);
			const auto working = std::filesystem::canonical(std::filesystem::current_path());
			Check(std::mismatch(repository.begin(), repository.end(), working.begin(), working.end()).first != repository.end(),
				"run tests from a scratch directory outside the repository");
			TimeInit();
			ShaderTranslationInit();
			g_VFS.Register("", new DirectoryReader(Path(std::string(argv[1]) + "/assets")));
			g_VFS.Register("metal-tests/", new DirectoryReader(Path(std::string(argv[1]) + "/GPU/metal4/tests")));
			g_Config.bShaderCache = false;
			g_Config.bReplaceTextures = false;
			g_Config.bSaveNewTextures = false;
			g_Config.iMultiSampleLevel = 0;
			std::string error;
			std::unique_ptr<Draw::DrawContext> draw(Draw::T3DCreateMetalContext(&error));
			Check(draw != nullptr, error);
			Check(draw->CreatePresets(), "Metal preset shaders");
			Memory::g_MemorySize = Memory::RAM_NORMAL_SIZE;
			Check(Memory::Init(Memory::MemMapSetupFlags::Default), "guest memory");
			int failures = 0;
			auto run = [&](const char *name, auto test) {
				const char *filter = getenv("METAL_TEST_FILTER");
				if (filter && std::string(name).find(filter) == std::string::npos) {
					return;
				}
				try {
					test();
				} catch (const std::exception &error) {
					std::cerr << "FAIL " << name << ": " << error.what() << '\n';
					++failures;
				}
			};
			{
				CommandProbe commands(draw.get());
				gpu = &commands;
				run("dither", [&] { Dither(commands); });
				run("depth-mirrors", [&] { DepthMirrors(commands.Cache()); });
				run("swizzle", [&] { Swizzle(commands.Cache()); });
				run("hash-layout", [&] { HashLayout(commands.Cache()); });
				run("decode-bounds", [&] { DecodeBounds(commands.Cache()); });
				run("replacement-update", [&] { ReplacementUpdate(commands.Cache()); });
				run("curve-line-edges", [&] { CurveLineEdges(); });
				gpu = nullptr;
			}
			auto manager = (Metal::RenderManager *)draw->GetNativeObject(Draw::NativeObject::RENDER_MANAGER);
			run("multipass", [&] { Multipass(manager->Context()); });
			run("dependencies", [&] { DependencyAccounting(); });
			run("submission-dependencies", [&] { SubmissionDependencies(); });
			run("dependency-timing", [&] { DependencyTimings(false); });
			run("dependency-precise-timing", [&] { DependencyTimings(true); });
			run("library-reuse", [&] { LibraryReuse(); });
			run("cache-parse-gate", [&] { CacheParseGate(); });
			run("cache-integrity", [&] { CacheIntegrity(draw.get(), manager); });
			if (getenv("METAL_TEST_EXPECT_ARCHIVE_UNAVAILABLE")) {
				run("archive-unavailable", [&] { ArchiveUnavailable(draw.get(), manager); });
			} else {
				run("progressive-archives", [&] { ProgressiveArchives(draw.get(), manager); });
			}
			run("pipeline-identity", [&] { PipelineIdentity(draw.get(), manager); });
			run("dynamic-depth-attachments", [&] { DynamicDepthAttachments(); });
			run("primary-surface-replace", [&] { PrimarySurfaceChanges(0); });
			run("primary-surface-detach", [&] { PrimarySurfaceChanges(1); });
			run("primary-surface-resize", [&] { PrimarySurfaceChanges(2); });
			run("primary-resize-event", [&] { PrimarySurfaceChanges(3); });
			run("primary-lost-event", [&] { PrimarySurfaceChanges(4); });
			run("primary-got-event", [&] { PrimarySurfaceChanges(5); });
			run("primary-surface-rebind", [&] { PrimarySurfaceChanges(6); });
			run("periodic-cache-saves", [&] { PeriodicCacheSaves(); });
			run("drawable-readback-ordering", [&] { DrawableReadbackOrdering(); });
			run("multiple-outputs", [&] { MultipleOutputs(); });
			run("output-resize-no-wait", [&] { OutputRetirement(false); });
			run("output-destroy-no-wait", [&] { OutputRetirement(true); });
			run("output-presentation", [&] { OutputPresentation(draw.get()); });
			run("lazy-frame-commands", [&] { LazyFrameCommands(); });
			run("alpha-stencil-fetch", [&] { AlphaStencilFetch(draw.get(), manager); });
			run("deferred-clears", [&] { DeferredClears(draw.get(), manager); });
			run("copy-plane-reuse", [&] { CopyPlaneReuse(); });
			run("transfer-pipeline-identity", [&] { TransferPipelineIdentity(); });
			run("transfer-shader-reuse", [&] { UtilityShaderReuse(false); });
			run("alpha-stencil-shader-reuse", [&] { UtilityShaderReuse(true); });
			run("upload-pools", [&] { UploadPools(); });
			run("atlas-pressure", [&] { AtlasPressure(draw.get(), manager); });
			run("scaler-support", [&] { ScalingSupport(draw.get()); });
			run("flat-shading", [&] { FlatShading(draw.get(), manager, 0); });
			run("buffer-effects-changes", [&] { BufferEffectsChanges(draw.get()); });
			run("upstream-video-clamp", [&] { UpstreamVideoClamp(draw.get()); });
			run("upstream-framebuffers", [&] { UpstreamFramebuffers(draw.get()); });
			run("upstream-logic-draws", [&] { UpstreamLogicDraws(draw.get()); });
			run("near-clip-capacity", [&] { NearClipCapacity(); });
			run("flat-shading-msaa4", [&] { FlatShading(draw.get(), manager, 2); });
			Memory::Shutdown();
			Check(failures == 0, "regression groups failed: " + std::to_string(failures));
			std::cout << "PASS all Metal correctness regressions\n";
			return 0;
		} catch (const std::exception &error) {
			std::cerr << "FAIL " << error.what() << '\n';
			return 1;
		}
	}
}
