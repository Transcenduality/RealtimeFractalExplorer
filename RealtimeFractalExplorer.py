import sys, math
import pygame
import moderngl
import numpy as np
from pygame.locals import *

W, H = 1280, 720

VERT = """
#version 330 core
in  vec2 pos;
out vec2 uv;
void main(){ uv = pos; gl_Position = vec4(pos,0,1); }
"""

FRAG = """
#version 330 core
in  vec2  uv;
out vec4  fragColor;

uniform mat3  camRot;
uniform float fovScale;
uniform float aspect;
uniform float worldScale;
uniform vec3  camPosHi;
uniform vec3  camPosLo;
uniform float pixScale;
uniform int   iters;

vec2 de(vec3 p_local) {
    vec3 c_lo = p_local * worldScale + camPosLo;

    vec3 z_lo = c_lo;
    float dr   = 1.0;
    float trap = 1e9;

    for(int i = 0; i < iters; i++){
        vec3 z_g = camPosHi + z_lo;
        float r  = length(z_g);
        if(r > 4.0) break;

        float theta = acos(clamp(z_g.z/r,-1.0,1.0)) * 8.0;
        float phi   = atan(z_g.y, z_g.x) * 8.0;
        dr  = pow(r,7.0)*8.0*dr + 1.0;
        float zr = pow(r,8.0);

        vec3 z_next_g =
            zr*vec3(sin(theta)*cos(phi),
                    sin(theta)*sin(phi),
                    cos(theta))
            + camPosHi + c_lo;

        z_lo = z_next_g - camPosHi;
        trap = min(trap, r);
    }

    vec3 z_final = camPosHi + z_lo;
    float rz = length(z_final);
    float d  = 0.5*log(max(rz,1e-20))*rz / max(dr,1e-20);
    return vec2(d / worldScale, trap);
}

vec3 calcNormal(vec3 p) {
    vec2 e = vec2(1.0,-1.0) * 0.0004;
    return normalize(
        e.xyy*de(p+e.xyy).x + e.yyx*de(p+e.yyx).x +
        e.yxy*de(p+e.yxy).x + e.xxx*de(p+e.xxx).x);
}

float calcAO(vec3 p, vec3 n, float hitDist) {
    float ao   = 0.0;
    float w    = 0.5;
    float base = max(hitDist * 3.0, 1e-5);

    float s[8];
    s[0]=base*1.0; s[1]=base*2.2; s[2]=base*4.0; s[3]=base*7.0;
    s[4]=base*11.0;s[5]=base*16.0;s[6]=base*22.0;s[7]=base*30.0;

    for(int i=0;i<8;i++){
        float d = de(p + n*s[i]).x;
        ao += w * max(0.0, s[i] - d);
        w  *= 0.55;
    }

    float raw = clamp(1.0 - ao/(base*3.5), 0.0, 1.0);
    return raw*raw*raw;
}

void main(){

    vec3 rd = normalize(camRot * vec3(uv*vec2(aspect,1.0)*fovScale, 1.0));

    float t    = 0.0;
    float tMax = 10.0;
    bool  hit  = false;
    vec2  res;
    float steps = 0.0;

    for(int i = 0; i < 300; i++){
        res = de(rd * t);
        float eps = t * pixScale * 0.5;

        if(res.x < eps){
            hit = true;
            break;
        }

        t += res.x * 0.85;
        steps += 1.0;

        if(t > tMax) break;
    }

    vec3 col = vec3(0.0);

    if(hit){
        vec3  p  = rd * t;
        vec3  n  = calcNormal(p);
        float ao = calcAO(p, n, res.x);

        float facing  = clamp(dot(n, -rd), 0.0, 1.0);
        float fresnel = pow(1.0 - facing, 4.0);

        float trap     = clamp(1.0 - res.y, 0.0, 1.0);
        float veinMask = pow(trap, 6.0) * 3.0;

        vec3  base    = vec3(0.015, 0.03, 0.06) * ao;
        vec3  rimCol  = vec3(0.0, 0.85, 1.0);
        vec3  veinCol = vec3(0.0, 0.5, 0.9);

        float rimStr  = fresnel * sqrt(ao) * 1.8;

        col = base + rimCol * rimStr + veinCol * veinMask * ao * 0.4;
    }

    /* SPACE GAS — affects empty space too */
    float gas = 0.001 * steps * exp(0.1 * steps);
    col += vec3(0.0,1.0,1.0) * gas;

    col = (col*(2.51*col+2)) / (col*(2.43*col+2)+5);
    fragColor = vec4(pow(max(col,vec3(0.0)), vec3(0.4545)), 1.0);
}
"""


def cpu_fractal(pos, iters=20):
    zx,zy,zz = pos
    ox,oy,oz = pos
    dr = 1.0
    for _ in range(iters):
        r = math.sqrt(zx*zx+zy*zy+zz*zz+1e-25)
        if r > 4.0: break
        theta = math.acos(max(-1.0,min(1.0,zz/r)))*8.0
        phi   = math.atan2(zy,zx)*8.0
        dr    = 8.0*pow(r,7.0)*dr+1.0
        zr    = pow(r,8.0)
        zx,zy,zz = (zr*math.sin(theta)*math.cos(phi)+ox,
                    zr*math.sin(theta)*math.sin(phi)+oy,
                    zr*math.cos(theta)+oz)
    rz = math.sqrt(zx*zx+zy*zy+zz*zz)
    return 0.5*math.log(max(rz,1e-25))*rz/max(dr,1e-25)


class Camera:
    def __init__(self): self.reset()

    def reset(self):
        self.pos        = np.array([0.0,0.0,-3.0], dtype=np.float64)
        self.R          = np.eye(3, dtype=np.float64)
        self.zoom_level = 0.0
        self.base_speed = 1.5

    def look(self, dx, dy):
        for axis,ang in [(self.R[:,1],dx*0.002),(self.R[:,0],dy*0.002)]:
            u = axis/(np.linalg.norm(axis)+1e-15)
            c,s = math.cos(ang),math.sin(ang)
            K = np.array([[0,-u[2],u[1]],[u[2],0,-u[0]],[-u[1],u[0],0]])
            self.R = (c*np.eye(3)+s*K+(1-c)*np.outer(u,u))@self.R

    def move(self, keys, dt):
        world_scale = math.pow(10, -self.zoom_level)
        move_dir    = np.zeros(3, dtype=np.float64)
        if keys[K_w]: move_dir += self.R[:,2]
        if keys[K_s]: move_dir -= self.R[:,2]
        if keys[K_d]: move_dir += self.R[:,0]
        if keys[K_a]: move_dir -= self.R[:,0]
        if np.linalg.norm(move_dir) > 0:
            move_dir /= np.linalg.norm(move_dir)

        speed = self.base_speed * dt
        if keys[K_LSHIFT]: speed *= 3.0

        dist_local = cpu_fractal(self.pos) / world_scale
        safe_step  = min(speed, max(dist_local*0.4, 0.0))
        self.pos  += move_dir * safe_step * world_scale


def main():
    pygame.init()
    pygame.display.set_mode((W,H), DOUBLEBUF|OPENGL)
    ctx   = moderngl.create_context()
    quad  = ctx.buffer(np.array([-1,-1,1,-1,-1,1,1,1],dtype=np.float32))
    prog  = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
    vao   = ctx.vertex_array(prog,[(quad,"2f","pos")])
    cam   = Camera()
    clock = pygame.time.Clock()
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)

    while True:
        dt = clock.tick(60)/1000.0
        for ev in pygame.event.get():
            if ev.type==QUIT or (ev.type==KEYDOWN and ev.key==K_ESCAPE):
                pygame.quit(); sys.exit()
            if ev.type==MOUSEMOTION: cam.look(ev.rel[0],ev.rel[1])
            if ev.type==MOUSEWHEEL:
                cam.base_speed = float(np.clip(cam.base_speed+ev.y*0.2,0.1,10.0))
            if ev.type==KEYDOWN and ev.key==K_r: cam.reset()

        cam.move(pygame.key.get_pressed(), dt)
        w_scale = math.pow(10, -cam.zoom_level)

        # Dekker split: encode float64 camera position as two float32 vectors.
        # hi = position rounded to float32.
        # lo = the residual (what float32 dropped) — carries the lost precision.
        # Together they represent a 48-bit effective mantissa instead of 24-bit.
        hi = cam.pos.astype(np.float32)
        lo = (cam.pos - hi.astype(np.float64)).astype(np.float32)

        prog["camPosHi"].value  = tuple(hi)
        prog["camPosLo"].value  = tuple(lo)
        prog["worldScale"].value= float(w_scale)
        prog["camRot"].value    = cam.R.T.astype(np.float32).flatten().tolist()
        prog["fovScale"].value  = 0.577
        prog["aspect"].value    = W/H
        prog["pixScale"].value  = 1.0/H
        prog["iters"].value     = int(max(20, min(20 + (cam.zoom_level**1.6)*8, 500)))

        vao.render(moderngl.TRIANGLE_STRIP)
        pygame.display.flip()
        pygame.display.set_caption(
            f"Mandelbulb  |  Zoom: 10^{cam.zoom_level:.1f}"
            f"  |  Speed: {cam.base_speed:.1f}"
            f"  |  [WASD] move [R] reset  [Shift] fast"
        )


if __name__ == "__main__":
    main()
