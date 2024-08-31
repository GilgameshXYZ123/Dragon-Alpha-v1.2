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

/**
 * Y = Tanh(A*X + B).
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Affine_Tanh extends Affine {
    private static final long serialVersionUID = 1L;
    
    public Affine_Tanh(boolean inplace, int... feature_dim) {
        super(inplace, feature_dim);
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    @Override public Affine_Tanh weight(Tensor weight) { set_weight(weight); return this; }
    @Override public Affine_Tanh bias(Tensor bias) { set_bias(bias); return this; }
    
    @Override
    protected InlineAffine_Tanh<?> create_unit_core() {
        return new InlineAffine_Tanh<>(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAffine_Tanh">
    public static class InlineAffine_Tanh<T extends Affine_Tanh> extends InlineAffine<T> 
    {
        public InlineAffine_Tanh(T unit) { super(unit); }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation"> 
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.affine_tanh(inplace, X, ut.A.ts(), ut.B.ts());
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            Tensor deltaA = null, deltaB = null, deltaX = null;
            int gc_count = 0;
        
            if(ut.A.need_grads() || ut.B.need_grads()) {//A.need_grads = B.need_grads = true
                Tensor[] grads = (is_holdY()? //V1: Y is not changed / V2: X is not changed
                        eg.affine_tanh_deltaAB_v1(deltaY, holdY(), ut.A.ts(), ut.B.ts()) :
                        eg.affine_tanh_deltaAB_v2(deltaY, holdX(), ut.A.ts(), ut.B.ts()));
                
                deltaA = grads[0]; 
                deltaB = grads[1];
                if(ut.A.need_grads()) { ut.A.grads().add(deltaA); gc_count++; } else deltaA.remote_delete();
                if(ut.B.need_grads()) { ut.B.grads().add(deltaB); gc_count++; } else deltaB.remote_delete();
            }
        
            if(backward_grads) {
                deltaX = (is_holdY()? //V1: Y is not changed / V2: X is not changed
                        eg.affine_tanh_deltaX_v1(false, deltaY, holdY(), ut.A.ts()) :
                        eg.affine_tanh_deltaX_v2(false, deltaY, holdX(), ut.A.ts(), ut.B.ts()));

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
