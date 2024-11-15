uniform sampler2D u_texture; // 0

varying vec4 v_color;
varying vec2 v_texcoord;
uniform vec2 mapsize;

uniform float rt_w; // render target width
uniform float rt_h; // render target height
float vx_offset= 0.7;

float offset[3] = float[]( 0.0, 1.3846153846, 3.2307692308 );
float weight[3] = float[]( 0.2270270270, 0.3162162162, 0.0702702703 );

void main() 
{ 
  vec3 tc = vec3(1.0, 0.0, 0.0);
  if (gl_TexCoord[0].x<(vx_offset-0.01))
  {
    vec2 uv = gl_TexCoord[0].xy;
    tc = v_color * weight[0];
    for (int i=1; i<3; i++) 
    {
      tc += texture2D(u_texture, v_texcoord + vec2(0.0, offset[i])/rt_h).rgb 
              * weight[i];
      tc += texture2D(u_texture, v_texcoord - vec2(0.0, offset[i])/rt_h).rgb 
             * weight[i];
    }
  }
  else if (gl_TexCoord[0].x>=(vx_offset+0.01))
  {
    tc = v_color;
  }
  gl_FragColor = vec4(tc, 1.0);
}