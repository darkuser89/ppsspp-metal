// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "Common/GPU/Metal/MetalGraphicsContext.h"
#include "Common/GPU/Metal/MetalRenderManager.h"
#include "Common/GPU/Metal/thin3d_metal.h"

bool MetalIsAvailable() {
	if (@available(macOS 13.0, iOS 16.0, *)) {
		id<MTLDevice> device = MTLCreateSystemDefaultDevice();
		return device && [device supportsFamily:MTLGPUFamilyMetal3];
	}
	return false;
}

static Metal::RenderManager *Manager(Draw::DrawContext *draw) {
	return draw ? (Metal::RenderManager *)draw->GetNativeObject(Draw::NativeObject::RENDER_MANAGER) : nullptr;
}

MetalGraphicsContext::~MetalGraphicsContext() {
	ShutdownAPI();
}

bool MetalGraphicsContext::InitAPI(void *wnd, std::string *deviceName, std::string *error) {
	error->clear();
	if (draw_) {
		*error = "Metal graphics context is already initialized";
		return false;
	}
	draw_ = Draw::T3DCreateMetalContext(error);
	if (!draw_) {
		return false;
	}
	if (deviceName) {
		*deviceName = draw_->GetDeviceCaps().deviceName;
	}
	return true;
}

bool MetalGraphicsContext::InitSurface(WindowSystem winsys, void *data1, void *data2, std::string *error) {
	error->clear();
	if (!draw_ || winsys != WINDOWSYSTEM_METAL_EXT || !data1 ||
		![(__bridge id)data1 isKindOfClass:[CAMetalLayer class]]) {
		*error = "Metal surface requires an initialized context and a CAMetalLayer";
		return false;
	}
	if (!Manager(draw_)->SetSurface((__bridge CAMetalLayer *)data1, error)) {
		return false;
	}
	if (!presetsCreated_) {
		if (!draw_->CreatePresets()) {
			*error = "Failed to compile Metal preset shaders";
			draw_->DestroyPresets();
			std::string ignored;
			Manager(draw_)->SetSurface(nil, &ignored);
			return false;
		}
		presetsCreated_ = true;
	}
	Resize();
	return true;
}

void MetalGraphicsContext::ShutdownSurface() {
	if (draw_) {
		std::string error;
		Manager(draw_)->SetSurface(nil, &error);
	}
}

void MetalGraphicsContext::ShutdownAPI() {
	ShutdownSurface();
	delete draw_;
	draw_ = nullptr;
	presetsCreated_ = false;
}

void MetalGraphicsContext::Resize() {
	if (draw_) {
		Manager(draw_)->ResizeSurface();
	}
}

void MetalGraphicsContext::Poll() {
	Resize();
}

void *MetalGraphicsContext::GetAPIContext() {
	return draw_ ? (void *)&Manager(draw_)->Context() : nullptr;
}
