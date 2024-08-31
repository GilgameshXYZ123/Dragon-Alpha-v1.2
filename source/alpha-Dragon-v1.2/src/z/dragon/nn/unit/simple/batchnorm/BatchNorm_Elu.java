/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.batchnorm;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 * Batch Normalization.
 * Y = elu(batchNorm(X, A, B))
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class BatchNorm_Elu extends BatchNorm {
    private static final long serialVersionUID = 1L;
    
    protected float alpha, k;

    public BatchNorm_Elu(boolean inplace, boolean affine, 
            float beta1, float beta2, float eps, 
            float alpha, float negative_slope,
            int... feature_dim) 
    {
        super(inplace, affine,
                beta1, beta2, eps,
                feature_dim);
        this.alpha = alpha;
        this.k = negative_slope;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final float alpha() { return k; }
    public BatchNorm_Elu alpha(float alpha) { this.alpha = alpha; return this; }
    
    public final float negative_slop() { return k; }
    public BatchNorm_Elu negative_slop(float negative_slope) { k = negative_slope; return this; }
    
    @Override public BatchNorm_Elu track_stats(boolean flag) { track_stats = flag; return this; }
      
    @Override public BatchNorm_Elu affine(boolean flag) { affine = flag; return this; }
    @Override public BatchNorm_Elu beta1(float beta1) { this.beta1 = beta1; return this; }
    @Override public BatchNorm_Elu beta2(float beta2) { this.beta2 = beta2; return this; }
    @Override public BatchNorm_Elu eps(float eps) { this.eps = eps; return this; }
    
    @Override public BatchNorm_Elu weight(Tensor weight) { set_weight(weight); return this; }
    @Override public BatchNorm_Elu bias(Tensor bias) { set_bias(bias); return this; }
    
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
                .append(eps).append("]");
        sb.append(", alpha = ").append(alpha);
        sb.append(", negative_slope = ").append(k);
        sb.append(" }");
    }
    
    @Override public BatchNorm_Elu train() { this.training = true; return this; }
    @Override public BatchNorm_Elu eval() { this.training = false; return this; }
    
    @Override
    protected InlineBatchNorm_Elu<?> create_unit_core() {
        return new InlineBatchNorm_Elu<>(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineBatchNorm_Elu">
    public static class InlineBatchNorm_Elu<T extends BatchNorm_Elu> extends InlineBatchNorm<T>
    {
        public InlineBatchNorm_Elu(T unit) { super(unit); }
        
        public final float alpha() { return ut.alpha; }
        public final float negative_slope() { return ut.k ; }
         
        //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
        @Override
        protected Tensor __forward_evaluate__(Engine eg, Tensor X, boolean inplace) {
            inplace = (inplace || ut.training);
            return (ut.affine?
                    eg.batchNorm_elu(inplace, X, ut.run_mean.ts(), ut.run_var.ts(), ut.eps, ut.A.ts(), ut.B.ts(), ut.alpha, ut.k) :
                    eg.batchNorm_elu(inplace, X, ut.run_mean.ts(), ut.run_var.ts(), ut.eps, ut.alpha, ut.k));
        }
        
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
                    eg.batchNorm_elu(inplace, X, ut.dX_mean, ut.dX_var, ut.eps, ut.A.ts(), ut.B.ts(), ut.alpha, ut.k) :
                    eg.batchNorm_elu(inplace, X, ut.dX_mean, ut.dX_var, ut.eps, ut.alpha, ut.k));
            if(ut.track_stats) Y.dual(()-> { ut.run_mean.c(); ut.run_var.c(); });
            return Y;
        }
        //</editor-fold>
    
        //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
        @Override
        protected Tensor __backward_no_affine__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!backward_grads) return null;
            return (is_holdY() ? //V1: Y is not changed / V2: X is not changed
                    eg.batchNorm_elu_deltaX_v1(grad_inplace, deltaY, ut.alpha, ut.k, holdY(), ut.dX_var, ut.eps) :
                    eg.batchNorm_elu_deltaX_v2(grad_inplace, deltaY, ut.alpha, ut.k, holdX(), ut.dX_mean, ut.dX_var, ut.eps));
        }
    
        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            if(!ut.affine) return __backward_no_affine__(eg, deltaY, grad_inplace, backward_grads);//affine = false
        
            if(ut.A.need_grads() || ut.B.need_grads() || backward_grads) {
                Tensor[] grads = (is_holdY() ? //V1: Y is not changed / V2: X is not changed
                        eg.batchNorm_elu_gradients_v1(grad_inplace, deltaY, ut.alpha, ut.k, holdY(), 
                                ut.dX_var, ut.eps, ut.A.ts(), ut.B.ts()) :
                        eg.batchNorm_elu_gradients_v2(grad_inplace, deltaY, ut.alpha, ut.k, holdX(),
                                ut.dX_mean, ut.dX_var, ut.eps, ut.A.ts(), ut.B.ts()));

                Tensor deltaX = grads[0];
                Tensor deltaA = grads[1];
                Tensor deltaB = grads[2];
                
                if(ut.A.need_grads()) ut.A.grads().add(deltaA); else deltaA.remote_delete();
                if(ut.B.need_grads()) ut.B.grads().add(deltaB); else deltaB.remote_delete();
                
                if(backward_grads) { ut.A.ts().follow(deltaX); ut.B.ts().follow(deltaX); return deltaX; }
                else { deltaX.remote_delete(); return null; }
            }
           
            return null;
        }
        //</editor-fold>
    }
    //</editor-fold>
}