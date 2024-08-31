/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.batchnorm;

import z.dragon.nn.unit.simple.batchnorm.global.GlobalSqBatchNorm;
import z.dragon.engine.Tensor;
import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor.TensorSet;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 * Batch Normalization.
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class SqBatchNorm extends GlobalSqBatchNorm {
    private static final long serialVersionUID = 1L;
    
    protected boolean track_stats = true;
    
    transient protected Tensor dX_mean;
    transient protected Tensor dX_sqmean;
    
    public SqBatchNorm(boolean inplace, boolean affine,
            float beta1, float beta2, float eps,
            int... feature_dim) 
    {
        super(inplace, affine,
              beta1, beta2, eps,
              feature_dim);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public boolean track_stats() { return track_stats; }
    public SqBatchNorm track_stats(boolean flag) { track_stats = flag; return this; }
    
    @Override public SqBatchNorm affine(boolean flag) { affine = flag; return this; }
    @Override public SqBatchNorm beta1(float beta1) { this.beta1 = beta1; return this; }
    @Override public SqBatchNorm beta2(float beta2) { this.beta2 = beta2; return this; }
    @Override public SqBatchNorm eps(float eps) { this.eps = eps; return this; }
    
    public Tensor mean() { return dX_mean; }
    public Tensor sqmean() { return dX_sqmean; }
    
    @Override public SqBatchNorm weight(Tensor weight) { set_weight(weight); return this; }
    @Override public SqBatchNorm bias(Tensor bias) { set_bias(bias); return this; }
    
    @Override public SqBatchNorm run_mean(Tensor mean) { set_run_mean(mean); return this; }
    @Override public SqBatchNorm run_sqmean(Tensor sqmean) { set_run_sqmean(sqmean); return this; }
    
    @Override public SqBatchNorm train() { this.training = true; return this; }
    @Override public SqBatchNorm eval() { this.training = false; return this; }
    
    @Override 
    public void append(String pre, StringBuilder sb) { 
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", training = ").append(training);
        sb.append(", affine = ").append(affine);
        sb.append(", track_stats = ").append(track_stats);
        sb.append(", features = ").append(features);
        sb.append(", param_dim = ["); Vector.append(sb, param_dim); sb.append("] ");
        sb.append(", [beta1, beta2, eps] = [")
                .append(beta1).append(", ")
                .append(beta2).append(", ")
                .append(eps).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    @Override
    protected InlineSqBatchNorm<?> create_unit_core() {
        if(ucm.count() >= 1) throw new RuntimeException("Normalization Cores can only be called once in a cycle");
        return new InlineSqBatchNorm<>(this);
    }
    
    @Override
    public void variables(TensorSet set) {
        super.variables(set);
        set.add(dX_mean, dX_sqmean);
    }
    
    @Override
    public void gc() {
        super.gc();
        if(dX_mean != null) { dX_mean.delete(); dX_mean = null; }
        if(dX_sqmean != null) { dX_sqmean.delete(); dX_sqmean = null;  }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineSqBatchNorm">
    public static class InlineSqBatchNorm<T extends SqBatchNorm> extends InlineGlobalSqBatchNorm<T>
    {
        public InlineSqBatchNorm(T unit) { super(unit); }
     
        public final boolean track_stats() { return ut.track_stats; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace)  {
            if(!ut.training) return __forward_evaluate__(eg, X, inplace);

            //====[Stage1: update run_mean and run_sqmean]==========================
            Tensor[] stats = eg.field_mean_sqmean(X, ut.features);
            ut.dX_mean   = stats[0]; 
            ut.dX_sqmean = stats[1];
            
            if(ut.track_stats) {
                //Update: run_mean = a1*run_mean + a2*dX_mean
                //[1] a1 = alpha * (1 - alpha^(t-1)) / (1 - alpha^t)
                //[2] a2 = (1 - alpha) / (1 - alpha^t)
                double last_correct_beta1 = 1 - ut.expBeta1; 
                ut.expBeta1 *= ut.beta1;
                double corrrect_beta1 = 1 - ut.expBeta1;
                float a1 = (float) (ut.beta1 * last_correct_beta1 / corrrect_beta1);
                float a2 = (float) ((1.0 - ut.beta1) / corrrect_beta1);
                eg.add(true, a1, ut.run_mean.ts(), a2, ut.dX_mean.c());//inplace: run_mean
            
                //Update: run_sqmean = b1*run_sqmean + b2*dX_sqmean
                //[1] b1 = beta * (1 - beta^(t-1)) / (1 - beta^t)
                //[2] b2 = (1 - beta)/ (1 - beta^t)
                double last_correct_beta2 = 1 - ut.expBeta2;
                ut.expBeta2 *= ut.beta2;
                double correct_beta2 = 1 - ut.expBeta2;
                float b1 = (float) (ut.beta2 * last_correct_beta2 / correct_beta2);
                float b2 = (float) ((1.0 - ut.beta2) / correct_beta2);
                eg.add(true, b1, ut.run_sqmean.ts(), b2, ut.dX_sqmean.c());//inplace: run_sqmean
            }
            else { ut.dX_mean.c(); ut.dX_sqmean.c(); }
        
            //====Stage2: Batch Normalization=======================================
            Tensor Y = (ut.affine?
                    eg.sqBatchNorm(inplace, X, ut.dX_mean, ut.dX_sqmean, ut.eps, ut.A.ts(), ut.B.ts()):
                    eg.sqBatchNorm(inplace, X, ut.dX_mean, ut.dX_sqmean, ut.eps));
            if(ut.track_stats) Y.dual(()-> { ut.run_mean.c(); ut.run_sqmean.c(); });
            return Y;
        }
        //</editor-fold>
  
        //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
        @Override
        protected Tensor __backward_no_affine__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            return (is_holdX() ? //V2: X is not changed / V1: Y is not changed;
                    eg.sqBatchNorm_deltaX_v2(grad_inplace, deltaY, holdX(), ut.dX_mean, ut.dX_sqmean, ut.eps): 
                    eg.sqBatchNorm_deltaX_v1(grad_inplace, deltaY, holdY(), ut.dX_mean, ut.dX_sqmean, ut.eps));
        }
    
        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!ut.affine) return __backward_no_affine__(eg, deltaY, grad_inplace, backward_grads);//affine = false
        
            //======[integrally finds gradients for {A, B, X} ]=================
            if(ut.A.need_grads() && ut.B.need_grads() && backward_grads) {
                Tensor[] grads = (is_holdX() ? //V2: X is not changed / V1: Y is not changed 
                        eg.sqBatchNorm_gradients_v2(grad_inplace, deltaY, holdX(), ut.dX_mean, ut.dX_sqmean, ut.eps, ut.A.ts()) : 
                        eg.sqBatchNorm_gradients_v1(grad_inplace, deltaY, holdY(), ut.dX_mean, ut.dX_sqmean, ut.eps, ut.A.ts(), ut.B.ts()));
            
                Tensor deltaX = grads[0];
                ut.A.grads().add(grads[1]);
                ut.B.grads().add(grads[2]);

                ut.A.ts().follow(deltaX);//When compute deltaX, A can't be changed
                if(is_holdY()) ut.B.ts().follow(deltaX);//When holdY & compute deltaX, B can't be changed
                return deltaX;
            }
        
            //======[separately finds gradients for {A, B, X} ]=================
            Tensor deltaA = null, deltaB = null, deltaX = null; 
            int gc_count = 0;

            if(ut.A.need_grads() && ut.B.need_grads()) {
                Tensor[] grads = (is_holdY() ? //V1: Y is not changed / V2: X is not changed
                        eg.sqBatchNorm_deltaAB_v1(deltaY, holdY(), ut.A.ts(), ut.B.ts()) :
                        eg.sqBatchNorm_deltaAB_v2(deltaY, holdX(), ut.dX_mean, ut.dX_sqmean, ut.eps));
                
                ut.A.grads().add(deltaA = grads[0]);
                ut.B.grads().add(deltaB = grads[1]);
                if(grad_inplace) gc_count += 2;
            }
            else if(ut.A.need_grads()) {//B.need_grads = false
                deltaA = (is_holdY() ? //V1: Y is not changeds / V2: X is not changed
                        eg.sqBatchNorm_deltaA_v1(deltaY, holdY(), ut.A.ts(), ut.B.ts()) : 
                        eg.sqBatchNorm_deltaA_v2(deltaY, holdX(), ut.dX_mean, ut.dX_sqmean, ut.eps));
                ut.A.grads().add(deltaA);
                if(grad_inplace) gc_count++;
            }
            else if(ut.B.need_grads()) {//A.need_grads = false
                deltaB = eg.field_sum(deltaY, ut.features);
                ut.B.grads().add(deltaB);
                if(grad_inplace) gc_count++;
            }
        
            if(backward_grads) {  
                deltaX = (is_holdX() ? //V2: X is not changed / V1: Y is not changed
                        eg.sqBatchNorm_deltaX_v2(false, deltaY, holdX(), ut.dX_mean, ut.dX_sqmean, ut.eps, ut.A.ts()) :
                        eg.sqBatchNorm_deltaX_v1(false, deltaY, holdY(), ut.dX_mean, ut.dX_sqmean, ut.eps, ut.A.ts(), ut.B.ts()));
            
                ut.A.ts().follow(deltaX);//When compute deltaX, A can't be changed
                if(is_holdY()) ut.B.ts().follow(deltaX);//When holdY & compute deltaX, B can't be changed
                if(grad_inplace) gc_count++;
            }
            
            if(gc_count != 0) {//when [deltaA, deltaB, deltaX] are cauculated, [deltaY, dX_mean, dX_sqmean] are not needed
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