/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.math2;

import z.dragon.engine.Engine;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class LogSoftmax extends SimpleInplaceFunction
{
    private static final long serialVersionUID = 1L;
    
    protected int features;
    
    public LogSoftmax(boolean inplace, int features) {
        super(inplace);
        this.features = features;
    }
    
    //<editor-fold defaultstate="collapsed" desc="functions">
    public final int features() { return features; }
    public LogSoftmax features(int features) { this.features = features; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", features = ").append(features).append(" }");
    }
    
    @Override
    protected InlineLogSoftmax create_unit_core() {
        return new InlineLogSoftmax(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineLogSoftmax">
    public static class InlineLogSoftmax extends SimpleInplaceInline<LogSoftmax>
    {
        public InlineLogSoftmax(LogSoftmax unit) { super(unit); }
        
        public final int features() { return ut.features; }

        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.log_softmax(inplace, X, ut.features);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads?
                    eg.log_softmax_deltaX(grad_inplace, deltaY, holdY(), ut.features):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
