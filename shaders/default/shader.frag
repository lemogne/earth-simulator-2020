#version 130
uniform float brightness;
uniform sampler2D u_texture;
uniform sampler2D u_biome;
uniform vec2 mapsize;

varying vec4 v_color;
varying vec2 v_texcoord;
varying vec2 v_biomecoord;

void main() {
	float saturation = (brightness + 0.2) / 1.2;
	vec4 tex = texture2D(u_texture, v_texcoord / mapsize);
	tex *= (v_biomecoord.x > -20000) ? texture2D(u_biome, v_biomecoord / 32768.0f) : vec4(1.0f);
	float av = dot(tex, vec4(0.1, 0.5, 0.4, 0));
    gl_FragColor = v_color * (tex * saturation + (vec4(av, av, av, 0) + vec4(0, 0, 0, 1) * tex) * (1 - saturation));
}