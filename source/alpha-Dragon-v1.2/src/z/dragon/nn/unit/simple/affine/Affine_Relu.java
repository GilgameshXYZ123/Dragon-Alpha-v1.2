/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.affine;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Train2Eval;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 * Y = Relu(A*X + B).
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Affine_Relu extends Affine implements Train2Eval {
    private static final long serialVersionUID = 1L;
    
    public Affine_Relu(boolean inplace, int... feature_dim) {
        super(inplace, feature_dim);
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    @Override public Affine_Relu weight(Tensor weight) { set_weight(weight); return this; }
    @Override public Affine_Relu bias(Tensor bias) { set_bias(bias); return this; }
    
    protected boolean training = true;
    @Override public boolean training() { return training; }
    @Override public Affine_Relu train() { training = true; return this; }
    @Override public Affine_Relu eval() { training = false; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", traing = ").append(training);
        sb.append(", features = ").append(features);
        sb.append(", param_dim = ["); Vector.append(sb, param_dim); sb.append("] ");
        sb.append(" }");
    }
    
    @Override
    protected InlineAffine_Relu<?> create_unit_core() {
        return new InlineAffine_Relu<>(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineAffine_Relu">
    public static class InlineAffine_Relu<T extends Affine_Relu> extends InlineAffine<T> 
    {
        public InlineAffine_Relu(T unit) { super(unit); }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation"> 
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            if(ut.training) inplace = false;
            return eg.affine_relu(inplace, X, ut.A.ts(), ut.B.ts());
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            Tensor deltaA = null, deltaB = null, deltaX = null;
            int gc_count = 0;
        
            if(ut.A.need_grads() || ut.B.need_grads()) {//A.need_grads = B.need_grads = true
                Tensor[] grads = eg.affine_relu_deltaAB_v2(deltaY,
                        holdX(),//V2: X is not changed, it's impossible to calculate gradeints in terms of y
                        ut.A.ts(), ut.B.ts());
                deltaA = grads[0];
                deltaB = grads[1];
                
                if(ut.A.need_grads()) { ut.A.grads().add(deltaA); gc_count++; } else deltaA.remote_delete();
                if(ut.B.need_grads()) { ut.B.grads().add(deltaB); gc_count++; } else deltaB.remote_delete();
            }
        
            if(backward_grads) {
                deltaX = eg.affine_relu_deltaX_v2(false, deltaY, 
                        holdX(),//V2: X is not changed, it's impossible to calculate gradeints in terms of y
                        ut.A.ts(), ut.B.ts());
                
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
