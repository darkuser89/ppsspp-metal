// Copyright (c) 2026- PPSSPP Project.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <string>

namespace Draw {
class DrawContext;

// Starts offscreen. The platform GraphicsContext may attach a CAMetalLayer
// through NativeObject::RENDER_MANAGER (Metal::RenderManager).
// No Vulkan instance or MoltenVK library is involved.
DrawContext *T3DCreateMetalContext(std::string *error);
}
