#ifndef __articulation_renderer_h__
#define ___articulation_renderer_h__

/**
 * @defgroup ArticulationRenderer ArticulationRenderer renderer
 * @brief BotVis Viewer renderer plugin
 * @include articulation-renderer/articulation_renderer.h
 *
 * TODO
 *
 * Linking: `pkg-config --libs articulation-renderer`
 * @{
 */

#include <lcm/lcm.h>

#include <bot_vis/bot_vis.h>
#include <bot_frames/bot_frames.h>

#ifdef __cplusplus
extern "C" {
#endif

void articulation_add_renderer_to_viewer(BotViewer* viewer, int priority,lcm_t* lcm, BotFrames * frames, const char * articulation_frame);

/**
 * @}
 */

#ifdef __cplusplus
}
#endif

#endif
