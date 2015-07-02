#include <stdio.h>
#include <stdlib.h>

#include <gdk/gdkkeysyms.h>

#include <bot_vis/bot_vis.h>

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#define RENDERER_NAME "WAVEFRONT"

#define PARAM_ROT_X "Rot X"
#define PARAM_ROT_Y "Rot Y"
#define PARAM_ROT_Z "Rot Z"
#define PARAM_SCALE "Scale"
#define PARAM_DX    "dX"
#define PARAM_DY    "dY"
#define PARAM_DZ    "dZ"

typedef struct _RendererWavefront {
    BotRenderer renderer;

    BotWavefrontModel *wavefront_model;
    int display_lists_ready;
    GLuint wavefront_dl;

    BotViewer          *viewer;
    BotGtkParamWidget *pw;
} RendererWavefront;

static void
_renderer_free (BotRenderer *super)
{
    RendererWavefront *self = (RendererWavefront*) super->user;
    if (self->wavefront_model)
        bot_wavefront_model_destroy(self->wavefront_model);
    free (self);
}

static void
draw_wavefront_model (RendererWavefront * self)
{
    glPushMatrix ();
    glEnable (GL_BLEND);
    glEnable (GL_RESCALE_NORMAL);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glShadeModel (GL_SMOOTH);
    glEnable (GL_LIGHTING);
    glCallList (self->wavefront_dl);
    glPopMatrix ();
}

static GLuint
compile_display_list (RendererWavefront * self, char * prefix, BotWavefrontModel * model)
{
    GLuint dl = glGenLists (1);
    glNewList (dl, GL_COMPILE);

    glEnable(GL_LIGHTING);
    bot_wavefront_model_gl_draw(model);
    glDisable(GL_LIGHTING);

    glEndList ();
    return dl;
}

static void 
_renderer_draw (BotViewer *viewer, BotRenderer *super)
{
    RendererWavefront *self = (RendererWavefront*) super->user;

    glEnable(GL_DEPTH_TEST);

    if (!self->display_lists_ready) {
        if (self->wavefront_model)
            self->wavefront_dl = compile_display_list (self, "wavefront", self->wavefront_model);
        self->display_lists_ready = 1;
    }

    glPushMatrix();

    double scale = bot_gtk_param_widget_get_double(self->pw, PARAM_SCALE);
    glScalef (scale, scale, scale);
    
    double rot_x = bot_gtk_param_widget_get_double(self->pw, PARAM_ROT_X);
    double rot_y = bot_gtk_param_widget_get_double(self->pw, PARAM_ROT_Y);
    double rot_z = bot_gtk_param_widget_get_double(self->pw, PARAM_ROT_Z);
    glRotatef(rot_z, 0, 0, 1);
    glRotatef(rot_y, 0, 1, 0);
    glRotatef(rot_x, 1, 0, 0);

    double dx = bot_gtk_param_widget_get_double(self->pw, PARAM_DX);
    double dy = bot_gtk_param_widget_get_double(self->pw, PARAM_DY);
    double dz = bot_gtk_param_widget_get_double(self->pw, PARAM_DZ);
    glTranslated(dx, dy, dz);

    if (self->display_lists_ready && self->wavefront_dl)
        draw_wavefront_model (self);

    glPopMatrix();
}

static void
on_param_widget_changed (BotGtkParamWidget *pw, const char *name, void *user)
{
    RendererWavefront *self = (RendererWavefront*) user;
    bot_viewer_request_redraw(self->viewer);
}

void setup_renderer_wavefront(BotViewer *viewer, int render_priority, const char *wavefront_fname);
void 
setup_renderer_wavefront(BotViewer *viewer, int render_priority, const char *wavefront_fname)
{
    RendererWavefront *self = (RendererWavefront*) calloc (1, sizeof (RendererWavefront));

    BotRenderer *renderer = &self->renderer;

    renderer->draw = _renderer_draw;
    renderer->destroy = _renderer_free;

    renderer->widget = bot_gtk_param_widget_new();
    renderer->name = RENDERER_NAME;
    renderer->user = self;
    renderer->enabled = 1;

    self->viewer = viewer;

    self->pw = BOT_GTK_PARAM_WIDGET(renderer->widget);

    double default_scale = 1.0;
    double default_rot_x  = 0;
    double default_rot_y = 0;
    double default_rot_z   = 0;
    double default_dx    = 0;
    double default_dy    = 0;
    double default_dz    = 0;

    if(wavefront_fname) {
        self->wavefront_model = bot_wavefront_model_create(wavefront_fname);
        double minv[3];
        double maxv[3];
        bot_wavefront_model_get_extrema(self->wavefront_model, minv, maxv);

        double span_x = maxv[0] - minv[0];
        double span_y = maxv[1] - minv[1];
        double span_z = maxv[2] - minv[2];

        double span_max = MAX(span_x, MAX(span_y, span_z));

        // leave the default scale and translations at default
        default_scale = 1.0;
        default_dx = 0.0;
        default_dy = 0.0;
        default_dz = 0.0;

        printf("WAVEFRONT extrema: [%f, %f, %f] [%f, %f, %f]\n", 
                minv[0], minv[1], minv[2],
                maxv[0], maxv[1], maxv[2]);
        printf("WAVEFRONT initial scale: %f\n", default_scale);
        printf("WAVEFRONT initial offsets: [%f, %f, %f]\n", default_dx, default_dy, default_dz);
    }

    bot_gtk_param_widget_add_double(self->pw, PARAM_SCALE, 
            BOT_GTK_PARAM_WIDGET_SPINBOX, 0, 99999999999, 0.00000001, default_scale);

    bot_gtk_param_widget_add_double(self->pw, PARAM_DX, 
            BOT_GTK_PARAM_WIDGET_SPINBOX, -9999999999, 99999999999, 0.01, default_dx);
    bot_gtk_param_widget_add_double(self->pw, PARAM_DY, 
            BOT_GTK_PARAM_WIDGET_SPINBOX, -9999999999, 99999999999, 0.01, default_dy);
    bot_gtk_param_widget_add_double(self->pw, PARAM_DZ, 
            BOT_GTK_PARAM_WIDGET_SPINBOX, -9999999999, 99999999999, 0.01, default_dz);

    bot_gtk_param_widget_add_double(self->pw, PARAM_ROT_X, 
            BOT_GTK_PARAM_WIDGET_SPINBOX, -180, 180, 0.1, default_rot_x);
    bot_gtk_param_widget_add_double(self->pw, PARAM_ROT_Y, 
            BOT_GTK_PARAM_WIDGET_SPINBOX, -180, 180, 0.1, default_rot_y);
    bot_gtk_param_widget_add_double(self->pw, PARAM_ROT_Z, 
            BOT_GTK_PARAM_WIDGET_SPINBOX, -180, 180, 0.1, default_rot_z);

    gtk_widget_show_all(renderer->widget);

    g_signal_connect (G_OBJECT (self->pw), "changed", 
                      G_CALLBACK (on_param_widget_changed), self);

    bot_viewer_add_renderer(viewer, &self->renderer, render_priority);
}
