#version 130
uniform float brightness;

varying vec4 v_color;
varying vec2 v_texcoord;

void main() {
	float shadow = length(gl_Normal.xyz);
	float shading = (dot(gl_Normal.xyz / shadow, vec3(0,1,0))+2.5)/3.5;
	v_color = vec4(gl_Color.rgb * shadow * (vec3(0.05, 0.075, 0.16) + brightness * vec3(0.95, 0.925, 0.84)) * shading, 1.0);
	v_texcoord = gl_MultiTexCoord0.xy;
	vec4 position = gl_Vertex;
	gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * position;
}