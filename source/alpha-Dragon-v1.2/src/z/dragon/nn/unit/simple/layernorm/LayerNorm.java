/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.layernorm;

import java.util.Arrays;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Parameter;
import z.dragon.engine.Tensor.TensorSet;
import z.dragon.nn.unit.simple.affine.Affine;
import z.util.lang.annotation.Passed;

/**
 * Layer Normalization
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class LayerNorm extends Affine {
    private static final long serialVersionUID = 1L;
    
    protected boolean affine;
    protected float eps;
    transient protected Tensor X_mean;
    transient protected Tensor X_sqmean;
    
    public LayerNorm(boolean inplace, boolean affine, float eps, int... feature_dim) {
        super(inplace, feature_dim);
        this.affine = affine;
        this.eps = eps;                                     
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final boolean affine() { return affine; }
    public LayerNorm affine(boolean flag) { affine = flag; return this; }
     
    public final float eps() { return eps; }
    public LayerNorm eps(float eps) { this.eps = eps; return this;}
    
    @Override public LayerNorm weight(Tensor weight) { super.weight(weight); return this; }
    @Override public LayerNorm bias(Tensor bias) { super.bias(bias); return this; }
    
    public Tensor mean() { return X_mean; }
    public Tensor sqmean() { return X_sqmean; }
    
    @Override 
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", [feature_num, param_dim] = [ ")
                .append(features).append(", ")
                .append(Arrays.toString(param_dim)).append(" ]");
        sb.append(", eps = ").append(eps).append(" }");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    protected InlineLayerNorm<?> create_unit_core() {
        if(ucm.count() >= 1) throw new RuntimeException("Normalization Cores can only be called once in a cycle");
        return new InlineLayerNorm<>(this);
    }
    
    @Override
    protected void __init__(Engine eg) {
        eg.delete(A, B);//params are inited to match the lastDim of input
        if(affine) {//perform indentity transform
            A = new Parameter(eg.ones(param_dim)).need_grads(true);
            B = new Parameter(eg.zeros(param_dim)).need_grads(true);
            Parameter.sync(A, B);
        }
    }
    
    @Override
    public void variables(TensorSet set) {
        super.variables(set);
        set.add(X_mean, X_sqmean);
    }
    
    @Override
    public void gc() {
        super.gc();
        if(X_mean != null) { X_mean.delete(); X_mean = null; }
        if(X_sqmean != null) { X_sqmean.delete(); X_sqmean = null; }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: BatchNorm_builtin">
    public static class InlineLayerNorm<T extends LayerNorm>  extends InlineAffine<T> 
    {
        public InlineLayerNorm(T unit) { super(unit); }
        
        public final boolean affine() { return ut.affine; }
        public final float eps() { return ut.eps; }
          
        //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            Tensor[] stats = eg.row_mean_sqmean(X, ut.features);
            ut.X_mean = stats[0];
            ut.X_sqmean = stats[1];
            ut.X_mean.c(); ut.X_sqmean.c();
        
            return (ut.affine? 
                    eg.layerNorm(inplace, X, ut.X_mean, ut.X_sqmean, ut.eps, ut.A.ts(), ut.B.ts()): 
                    eg.layerNorm(inplace, X, ut.X_mean, ut.X_sqmean, ut.eps));
        }
        //</editor-fold>
    
        //<editor-fold defaultstate="collapsed" desc="running-area: backwards-propagation">
        protected Tensor __backward_no_affine__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            return (is_holdX()? //V2: X is not changed) / V1: Y is not changed);
                    eg.layerNorm_deltaX_v2(grad_inplace, deltaY, holdX(), ut.X_mean, ut.X_sqmean, ut.eps): 
                    eg.layerNorm_deltaX_v1(grad_inplace, deltaY, holdY(), ut.X_mean, ut.X_sqmean, ut.eps));
        }
    
        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!ut.affine) return __backward_no_affine__(eg, deltaY, grad_inplace, backward_grads);//affine = false
            
            //======[separately finds gradients for {A, B, X} ]=================
            Tensor deltaA = null, deltaB = null, deltaX = null;
            int gc_count = 0;
        
            if(ut.A.need_grads() && ut.B.need_grads()) {//A.need_grads = B.need_grads = true
                Tensor[] grads = (is_holdY()?
                        eg.layerNorm_deltaAB_v1(deltaY, holdY(), ut.A.ts(), ut.B.ts())://V1: Y is not changed
                        eg.layerNorm_deltaAB_v2(deltaY, holdX(), ut.X_mean, ut.X_sqmean, ut.eps));//V2: X is not changed
            
                ut.A.grads().add(deltaA = grads[0]);
                ut.B.grads().add(deltaB = grads[1]);
                if(grad_inplace) gc_count += 2;
            }
            else if(ut.A.need_grads()) {//B.need_grads = false
                deltaA = (is_holdY() ? //V1: Y is not changed / V2: X is not changed
                        eg.layerNorm_deltaA_v1(deltaY, holdY(), ut.A.ts(), ut.B.ts()):
                        eg.layerNorm_deltaA_v2(deltaY, holdX(), ut.X_mean, ut.X_sqmean, ut.eps));
                ut.A.grads().add(deltaA);
                if(grad_inplace) gc_count++;
            }
            else if(ut.B.need_grads()) {//A.need_grads = false
                deltaB = eg.field_sum(deltaY, ut.features);
                ut.B.grads().add(deltaB);
                if(grad_inplace) gc_count++;
            }
        
            if(backward_grads) {
                deltaX = (is_holdX()?  //V1: Y is not changed / V2: X is not changed
                        eg.layerNorm_deltaX_v2(false, deltaY, holdX(), ut.X_mean, ut.X_sqmean, ut.eps, ut.A.ts()) :
                        eg.layerNorm_deltaX_v1(false, deltaY, holdY(), ut.X_mean, ut.X_sqmean, ut.eps, ut.A.ts(), ut.B.ts()));
            
                ut.A.ts().follow(deltaX);//When compute deltaX, A can't be changed
                if(is_holdY()) ut.B.ts().follow(deltaX);//When compute deltaX, B can't be changed
                if(grad_inplace) gc_count++;
            }
        
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
