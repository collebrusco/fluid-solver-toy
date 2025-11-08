#version 410 core

uniform sampler2D uTexslot;
out vec4 outColor;
in vec2 iUV;

void main(){
    vec4 c = texture(uTexslot, iUV);
    // if (c.r != 0.0) {c.r = 1.f;}
    outColor = vec4(c.r, 0.f, c.g, 1.f);
}
