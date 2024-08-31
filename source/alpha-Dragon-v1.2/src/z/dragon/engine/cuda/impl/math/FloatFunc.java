/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine.cuda.impl.math;

/**
 *
 * @author Gilgamesh
 */
public class FloatFunc 
{
    public static class FloatFuncConfig {
        public int type;
        public float[] params;
        private static final float[] null_params = new float[0];

        public FloatFuncConfig(int type, float... params) {
            this.type = type;
            this.params = (params != null ? params : null_params);
        }
    }
    
    public static int fcode_relu      = 0;
    public static int fcode_leakyRelu = 1;
    public static int fcode_elu       = 2;
    public static int fcode_softplus  = 3;
    public static int fcode_gelu      = 4;
    public static int fcode_sigmoid   = 5;
    public static int fcode_tanh      = 6;
    
    public static FloatFuncConfig relu() { return new FloatFuncConfig(fcode_relu);  }
    public static FloatFuncConfig leakyRelu(float k) { return new FloatFuncConfig(fcode_leakyRelu, k); }
    public static FloatFuncConfig elu(float alpha, float k) { return new FloatFuncConfig(fcode_elu, alpha, k); }
    public static FloatFuncConfig softplus() { return new FloatFuncConfig(fcode_softplus); }
    public static FloatFuncConfig gelu() { return new FloatFuncConfig(fcode_gelu); }
    public static FloatFuncConfig sigmoid() { return new FloatFuncConfig(fcode_sigmoid); }
    public static FloatFuncConfig tanh() { return new FloatFuncConfig(fcode_tanh);}
}
