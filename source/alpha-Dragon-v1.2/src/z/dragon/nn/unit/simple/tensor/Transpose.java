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
public class Transpose extends SimpleInplaceFunction
{
    private static final long serialVersionUID = 1L;
    
    protected int idx1;//dim Idx1
    protected int idx2;//dim Idx2
    
    public Transpose(boolean inplace, int dimIdx1, int dimIdx2) {
        super(inplace);
        this.idx1 = dimIdx1;
        this.idx2 = dimIdx2;
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public final int dimIdx1() { return idx1; }
    public Transpose dimIdx1(int dimIdx1) {  this.idx1 = dimIdx1; return this; }
    
    public final int dimIdx2() { return idx2; }
    public Transpose setDimIdx2(int dimIdx2) { this.idx2 = dimIdx2; return this; }
    
    public final int[] dimIdx() { return new int[]{ idx1, idx2 }; }
    public Transpose dimIdx(int dimIdx1, int dimIdx2) {
        this.idx1 = dimIdx1;
        this.idx2 = dimIdx2;
        return this;
    }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append(" { inplace = ").append(inplace());
        sb.append(", [dimIdx1, dimIdx2] = [").append(idx1).append(", ").append(idx2) .append("] }");
    }
    
    @Override
    protected InlineTranspose create_unit_core() {
        return new InlineTranspose(this);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="static class: InlineTranspose">
    public static class InlineTranspose extends SimpleInplaceInline<Transpose>
    {
        public InlineTranspose(Transpose unit) { super(unit); }
  
        public final int dimIdx1() { return ut.idx1; }
        public final int dimIdx2() { return ut.idx2; }
        public final int[] dimIdx() { return new int[]{ ut.idx1, ut.idx2 }; }
    
        //<editor-fold defaultstate="collapsed" desc="running-area">
        @Override
        protected Tensor __forward__(Engine eg, Tensor X, boolean inplace) {
            return eg.transpose(inplace, X, ut.idx1, ut.idx2);
        }

        @Override
        protected Tensor __backward__(Engine eg, Tensor deltaY,
                boolean grad_inplace, boolean backward_grads) {
            return (backward_grads? 
                    eg.transpose(grad_inplace, deltaY, ut.idx1, ut.idx2):
                    null);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
