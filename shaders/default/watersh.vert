#version 130
uniform float brightness;
uniform vec2 mapsize;

varying vec4 v_color;
varying vec2 v_texcoord;
varying vec2 v_biomecoord;

void main() {
	float shading = (dot(gl_Normal.xyz, vec3(0, 1, 0)) + 3.0) / 4.0;
	v_color = vec4((gl_Color.rgb * vec3(0.3, 0.4, 0.6) + vec3(0.4, 0.4, 0.4)) * (0.04 + brightness * 0.96) * shading, 1.0);
	v_texcoord = gl_MultiTexCoord0.xy;
	v_biomecoord = gl_MultiTexCoord1.xy;
	vec4 position = gl_Vertex;
	gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * position;
}