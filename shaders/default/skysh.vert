#version 130


uniform float brightness;

varying vec4 v_color;
varying vec2 v_texcoord;

void main()
{
    v_color = vec4(gl_Color.rgb, 1.0);
    v_texcoord = gl_MultiTexCoord0.xy;
    gl_Position = ftransform(); 
}