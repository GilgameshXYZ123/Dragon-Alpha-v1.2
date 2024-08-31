/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.affine;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 * Y = leakyRelu(A*X + B).
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Affine_LeakyRelu extends Affine {
    private static final long serialVersionUID = 1L;
    
    protected float k;
    
    public Affine_LeakyRelu(boolean inplace, float negative_slope, int... feature_dim) {
        super(inplace, feature_dim);
        this.k = negative_slope;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final float negative_slop() { return k; }
    public Affine_LeakyRelu negative_slop(float negative_slope) { k = negative_slope; return this; }
    
    @Override public Affine_LeakyRelu weight(Tensor weight) { set_weight(weight); return this; }
    @Override public Affine_LeakyRelu bias(Tensor bias) { set_bias(bias); return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", features = ").append(features);
        sb.append(", param_dim = ["); Vector.append(sb, param_dim); sb.append("]");
        sb.append(", negative_slope = ").append(k);
        sb.append(" }");
    }
     
    @Override
    protected InlineAffine_LeakyRelu<?> create_unit_core() {
        return new InlineAffine_LeakyRelu<>(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAffine_LeakyRelu">
    public static class InlineAffine_LeakyRelu<T extends Affine_LeakyRelu> extends InlineAffine<T> 
    {
        public InlineAffine_LeakyRelu(T unit) { super(unit); }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation"> 
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.affine_leakyRelu(inplace, X, ut.A.ts(), ut.B.ts(), ut.k);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            Tensor deltaA = null, deltaB = null, deltaX = null;
            int gc_count = 0;
        
            if(ut.A.need_grads() || ut.B.need_grads()) {//A.need_grads = B.need_grads = true
                Tensor[] grads = (is_holdY()? //V1: Y is not changed / V2: X is not changed
                        eg.affine_leakyRelu_deltaAB_v1(deltaY, ut.k, holdY(), ut.A.ts(), ut.B.ts()) :
                        eg.affine_leakyRelu_deltaAB_v2(deltaY, ut.k, holdX(), ut.A.ts(), ut.B.ts()));
                
                deltaA = grads[0]; 
                deltaB = grads[1];
                if(ut.A.need_grads()) { ut.A.grads().add(deltaA); gc_count++; } else deltaA.remote_delete();
                if(ut.B.need_grads()) { ut.B.grads().add(deltaB); gc_count++; } else deltaB.remote_delete();
            }
        
            if(backward_grads) {
                deltaX = (is_holdY()? //V1: Y is not changed / V2: X is not changed
                        eg.affine_leakyRelu_deltaX_v1(false, deltaY, ut.k, holdY(), ut.A.ts()) :
                        eg.affine_leakyRelu_deltaX_v2(false, deltaY, ut.k, holdX(), ut.A.ts(), ut.B.ts()));

                ut.A.ts().follow(deltaX);//When compute deltaX, A can't be changed
                ut.B.ts().follow(deltaX);//When compute deltaX, B can't be changed
                if(grad_inplace) gc_count++;
            }
        
            //the final gc process----------------------------------------------
            if(gc_count != 0) {//when deltaA, deltaB, deltaX are cauculated, deltaY is not needed
                CountGc gc = new CountGc(gc_count, deltaY);
                if(deltaA != null) { deltaA.dual(()-> { gc.countDown(); }).remote_sync(); }
                if(deltaB != null) { deltaB.dual(()-> { gc.countDown(); }).remote_sync(); }
                if(deltaX != null) { deltaX.dual(()-> { gc.countDown(); }); }
            }
        
            return deltaX;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
