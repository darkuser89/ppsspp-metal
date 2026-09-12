// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "GPU/Common/FramebufferManagerCommon.h"

class FramebufferManagerMetal : public FramebufferManagerCommon {
public:
	explicit FramebufferManagerMetal(Draw::DrawContext *draw);
};
