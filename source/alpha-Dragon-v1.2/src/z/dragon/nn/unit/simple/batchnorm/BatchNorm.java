/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.batchnorm;

import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.batchnorm.global.GlobalBatchNorm;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 * Batch Normalization.
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class BatchNorm extends GlobalBatchNorm {
    private static final long serialVersionUID = 1L;
    
    protected boolean track_stats = true;
   
    transient protected Tensor dX_mean;
    transient protected Tensor dX_var;
    
    public BatchNorm(boolean inplace, boolean affine,
            float beta1, float beta2, float eps, 
            int... feature_dim) 
    {
        super(inplace, affine,
                beta1, beta2, eps,
                feature_dim);
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public boolean track_stats() { return track_stats; }
    public BatchNorm track_stats(boolean flag) { track_stats = flag; return this; }
    
    @Override public BatchNorm affine(boolean flag) { affine = flag; return this; }
    @Override public BatchNorm beta1(float beta1) { this.beta1 = beta1; return this; }
    @Override public BatchNorm beta2(float beta2) { this.beta2 = beta2; return this; }
    @Override public BatchNorm eps(float eps) { this.eps = eps; return this; }
    
    public Tensor mean() { return dX_mean; }
    public Tensor var() { return dX_var; }
    
    @Override public BatchNorm weight(Tensor weight) { set_weight(weight); return this; }
    @Override public BatchNorm bias(Tensor bias) { set_bias(bias); return this; }
    
    @Override public BatchNorm run_mean(Tensor mean) { set_run_mean(mean); return this; }
    @Override public BatchNorm run_var(Tensor var) { set_run_var(var); return this; }
    
    @Override public BatchNorm train() { this.training = true; return this; }
    @Override public BatchNorm eval() { this.training = false; return this; }
    
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
    protected InlineBatchNorm<?> create_unit_core() {
        if(ucm.count() >= 1) throw new RuntimeException("Normalization Cores can only be called once in a cycle");
        return new InlineBatchNorm<>(this);
    }
    
    @Override
    public void variables(Tensor.TensorSet set) {
        super.variables(set);
        set.add(dX_mean, dX_var);
    }
    
    @Override
    public void gc() {
        super.gc();
        if(dX_mean != null) { dX_mean.delete(); dX_mean = null; }
        if(dX_var != null) { dX_var.delete(); dX_var = null; } 
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineBatchNorm">
    public static class InlineBatchNorm<T extends BatchNorm> extends InlineGlobalBatchNorm<T>
    {
        public InlineBatchNorm(T unit) { super(unit); }
         
        public boolean track_stats() { return ut.track_stats; }
        
        //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace)  {
            if(!ut.training) return __forward_evaluate__(eg, X, inplace);

            //====[Stage1: update run_mean and run_var]=========================
            Tensor[] stats = eg.field_var_mean(false, X, ut.features);
            ut.dX_var  = stats[0];//unbiased = false
            ut.dX_mean = stats[1];
        
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
            
                //Update: run_var = b1*run_var + b2*dX_var
                //[1] K = N / (N - 1), N = batch_size, for unbiased  estimation
                //[2] b1 = beta * (1-beta^(t-1)) / (1-beta^t)
                //[3] b2 = K * (1 - beta) / (1 - beta^t)
                int N = X.length() / ut.features;
                double K = N / (N - 1.0);
                double last_correct_beta2 = 1 - ut.expBeta2;
                ut.expBeta2 *= ut.beta2;
                double correct_beta2 = 1 - ut.expBeta2;
                float b1 = (float) (ut.beta2 * last_correct_beta2 / correct_beta2);
                float b2 = (float) (K * (1.0 - ut.beta2) / correct_beta2);
                eg.add(true, b1, ut.run_var.ts(), b2, ut.dX_var.c());//inplace: run_var
            }
            else { ut.dX_mean.c(); ut.dX_var.c(); }
        
            //====Stage2: Batch Normalization===================================
            Tensor Y = (ut.affine?
                    eg.batchNorm(inplace, X, ut.dX_mean, ut.dX_var, ut.eps, ut.A.ts(), ut.B.ts()):
                    eg.batchNorm(inplace, X, ut.dX_mean, ut.dX_var, ut.eps));
            if (ut.track_stats) Y.dual(()-> { ut.run_mean.c(); ut.run_var.c(); });
            return Y;
        }
        //</editor-fold>
    
        //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
        @Override
        protected Tensor __backward_no_affine__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            return (is_holdY() ? //V1: Y is not changed / V2: X is not changed
                    eg.batchNorm_deltaX_v1(grad_inplace, deltaY, holdY(), ut.dX_var, ut.eps):
                    eg.batchNorm_deltaX_v2(grad_inplace, deltaY, holdX(), ut.dX_mean, ut.dX_var, ut.eps));
        }
    
        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!ut.affine) return __backward_no_affine__(eg, deltaY, grad_inplace, backward_grads);//affine = false
        
            //======[integrally finds gradients for {A, B, X} ]=================
            if(ut.A.need_grads() && ut.B.need_grads() && backward_grads) {
                Tensor[] grads = (is_holdX() ? //V2: X is not changed / V1: Y is not changed
                        eg.batchNorm_gradients_v2(grad_inplace, deltaY, holdX(), ut.dX_mean, ut.dX_var, ut.eps, ut.A.ts()):
                        eg.batchNorm_gradients_v1(grad_inplace, deltaY, holdY(), ut.dX_var, ut.eps, ut.A.ts(), ut.B.ts()));
            
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
                        eg.batchNorm_deltaAB_v1(deltaY, holdY(), ut.A.ts(), ut.B.ts()):
                        eg.batchNorm_deltaAB_v2(deltaY, holdX(), ut.dX_mean, ut.dX_var, ut.eps));
                
                ut.A.grads().add(deltaA = grads[0]);
                ut.B.grads().add(deltaB = grads[1]);
                if(grad_inplace) gc_count += 2;
            }
            else if(ut.A.need_grads()) {//B.need_grads = false
                deltaA = (is_holdY()? 
                        eg.batchNorm_deltaA_v1(deltaY, holdY(), ut.A.ts(), ut.B.ts()) ://V1: Y is not changeds
                        eg.batchNorm_deltaA_v2(deltaY, holdX(), ut.dX_mean, ut.dX_var, ut.eps));//V2: X is not changed
                ut.A.grads().add(deltaA);
                if(grad_inplace) gc_count++;
            }
            else if(ut.B.need_grads()) {//A.need_grads = false
                deltaB = eg.field_sum(deltaY, ut.features);
                ut.B.grads().add(deltaB);
                if(grad_inplace) gc_count++;
            }
        
            if(backward_grads) { 
                deltaX = (is_holdX()? //V2: X is not changed / V1: Y is not changed);
                        eg.batchNorm_deltaX_v2(false, deltaY, holdX(), ut.dX_mean, ut.dX_var, ut.eps, ut.A.ts()):
                        eg.batchNorm_deltaX_v1(false, deltaY, holdY(), ut.dX_var, ut.eps, ut.A.ts(), ut.B.ts()));
            
                ut.A.ts().follow(deltaX);//When compute deltaX, A can't be changed
                if(is_holdY()) ut.B.ts().follow(deltaX);//When holdY & compute deltaX, B can't be changed
                if(grad_inplace) gc_count++;
            }
        
            if(gc_count != 0) {
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
