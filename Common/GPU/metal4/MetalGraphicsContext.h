// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "Common/GPU/GraphicsContext.h"

bool MetalIsAvailable();

// ShaderTranslationInit/Shutdown are owned by the application, as for thin3d's
// other contexts. InitSurface accepts the common CAMetalLayer window contract.
class MetalGraphicsContext final : public GraphicsContext {
public:
	~MetalGraphicsContext() override;
	bool InitAPI(void *wnd, std::string *deviceName, std::string *error) override;
	bool InitSurface(WindowSystem winsys, void *data1, void *data2, std::string *error) override;
	void ShutdownSurface() override;
	void ShutdownAPI() override;
	void Resize() override;
	void Poll() override;
	void *GetAPIContext() override;
	Draw::DrawContext *GetDrawContext() override { return draw_; }

private:
	Draw::DrawContext *draw_ = nullptr;
	bool presetsCreated_ = false;
};
