/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math;

import z.util.lang.Lang;

/**
 *
 * @author Gilgamesh
 */
public final class FP32Math {
    static { System.load(Lang.code_root_path(FP32Math.class) + "\\z\\util\\dll\\FP32Math.dll"); }
    
    private FP32Math() { }
    
    public static native float sinf(float x);
    public static native float cosf(float x);
    public static native float tanf(float x);
    
    public static native float asinf(float x);
    public static native float acosf(float x);
    public static native float atanf(float x);
    public static native float atan2f(float y, float x);//atanf(y/x)
    
    public static native float sinhf(float x);
    public static native float coshf(float x);
    public static native float tanhf(float x);
    
    public static native float expf(float x);
    public static native float expm1f(float x);
    public static native float logf(float x);
    public static native float log1pf(float x);
    public static native float powf(float x, float y);
    
    public static native float sqrtf(float x);
    public static native float cbrtf(float x);
    public static native float hypotf(float x, float y);
    
    public static native float ceilf(float x);
    public static native float floorf(float x);
}
