/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.tensor;

import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.simple.SimpleInplaceInline;
import z.dragon.nn.unit.simple.SimpleInplaceFunction;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Flatten extends SimpleInplaceFunction {
    private static final long serialVersionUID = 1L;
    
    transient protected int[] inDim;
    transient protected int[] outDim;
    
    public Flatten(boolean inplace) { super(inplace); }

    //<editor-fold defaultstate="collapsed" desc="functions">
    public final int[] in_dim() { return inDim; }
    public final int[] out_dim() { return outDim; }
    
    @Override
    protected InlineFlatten create_unit_core() {
        return new InlineFlatten(this);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineFlatten">
    public static class InlineFlatten extends SimpleInplaceInline<Flatten> 
    {
        transient protected int[] inDim;
        transient protected int[] outDim;

        public InlineFlatten(Flatten unit) { super(unit); }
      
        public final int[] in_dim() { return inDim; }
        public final int[] out_dim() { return outDim; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area: propagation">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            inDim = X.dim();
            outDim = new int[] { X.dim(0), X.length() / X.dim(0) };
            return eg.reshape(inplace, X, outDim);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads? 
                    eg.reshape(grad_inplace, deltaY, inDim) :
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
