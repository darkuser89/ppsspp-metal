// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#include "GPU/Metal/FramebufferManagerMetal.h"
#include "GPU/Common/PresentationCommon.h"

FramebufferManagerMetal::FramebufferManagerMetal(Draw::DrawContext *draw)
	: FramebufferManagerCommon(draw) {
	presentation_->SetLanguage(GLSL_VULKAN);
}
