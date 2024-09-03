/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2.dropout;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Elu_Dropout extends Dropout {
    private static final long serialVersionUID = 562781240350001L;
    
    protected float alpha, k;
    
    public Elu_Dropout(boolean inplace,
            float alpha, float negative_slope, 
            float nonzero_p) 
    {
        super(inplace, nonzero_p);
        this.alpha = alpha;
        this.k = negative_slope;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final float alpha() { return alpha; }
    public Elu_Dropout alpha(float alpha) { this.alpha = alpha; return this; }
    
    public final float negative_slop() { return k; }
    public Elu_Dropout negative_slop(float negative_slope) { k = negative_slope; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", training = ").append(training);
        sb.append(", alpha = ").append(alpha);
        sb.append(", negative_slope = ").append(k);
        sb.append(", nonzero_percent = ").append(p).append(" }");
    }
    
    @Override
    protected InlineLeakyElu_Dropout create_unit_core() {
        return new InlineLeakyElu_Dropout(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineLeakyElu_Dropout">
    public static class InlineLeakyElu_Dropout extends InlineDropout<Elu_Dropout> {
        public InlineLeakyElu_Dropout(Elu_Dropout unit) { super(unit); }
        
        public final float alpha() { return ut.alpha; }
        public final float negative_slop() { return ut.k; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            if(!ut.training) return eg.elu(inplace, X, ut.alpha, ut.k);//exp = (1/p)*p + 0*(1-p) = 1

            Tensor[] outs = eg.elu_dropout(X, ut.alpha, ut.k, ut.p);
            R = outs[1];
            return outs[0];
        }
        //</editor-fold>
    }
    //</editor-fold>
}
