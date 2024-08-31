/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.simple.blas;

import java.util.Arrays;
import z.dragon.common.state.State;
import z.dragon.engine.Counter.CountGc;
import z.dragon.engine.Engine;
import z.dragon.engine.Tensor;
import z.dragon.engine.Parameter;
import z.dragon.engine.Parameter.ParamMap;
import z.dragon.engine.Parameter.ParamSet;
import z.dragon.nn.unit.simple.SimpleUnit;
import z.dragon.nn.core.simple.SimpleCore;
import z.util.lang.annotation.Passed;

/**
 *
 * @author Gilgamesh
 */
@Passed("CudaFloat32Base")
public class Deconv2D extends SimpleUnit {
    private static final long serialVersionUID = 562781240120001L;
    
    protected int IC, OC;
    protected int FW;
    protected int sw;
    protected int pw;
    protected int OW;//reversed conv2D: (OW) -> (IW)
    protected boolean biased;
    
    transient protected Parameter W;
    transient protected Parameter B;
    
    public Deconv2D(boolean biasd,
            int in_channel, int out_channel, 
            int kernel_width, 
            int stride_width,
            int padding_width,
            int output_width) 
    {
        this.biased = biasd;
        this.IC = in_channel; this.OC = out_channel;
        this.FW = kernel_width;
        this.sw = stride_width;
        this.pw = padding_width;
        this.OW = output_width;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public boolean biased() { return biased; }
    public int out_channels() { return OC; }
    public int in_channels() { return IC; }
    
    public int[] kernel()  { return new int[]{ FW }; }
    public int[] stride()  { return new int[]{ sw }; }
    public int[] padding() { return new int[]{ pw }; }
    public int[] fans() { return new int[]{ FW*OC, FW*IC }; }//[fan_in, fan_out]
    
    public final int[] out_size() { return new int[] { OW }; }
    public Deconv2D out_size(int... out_size) {
        if(out_size == null || out_size.length != 1) throw new IllegalArgumentException();
        OW = out_size[1];
        return this;
    }
    
    public Parameter weight_param() { return W; }
    public Tensor weight() { return W.ts(); }
    public Deconv2D weight(Tensor weight) {
        if(Tensor.isNull(weight)) throw new NullPointerException("weight is null");
        if(!weight.dimEquals(IC, FW, OC)) throw new IllegalArgumentException(String.format(
                "%s : weight.dim { got %s } != [IC, FW, OC] = [%d, %d, %d]", 
                name, Arrays.toString(weight.dim()), IC, FW, OC));
        if(W != null) W.delete(); 
        W.tensor(weight);
        return this;
    }
    
    public Parameter bias_param() { return B; }
    public Tensor bias() { return B.ts(); }
    public Deconv2D bias(Tensor bias) {
        if(Tensor.isNull(bias)) throw new NullPointerException("bias is null");
        if(!bias.dimEquals(IC)) throw new IllegalArgumentException(String.format(
                "%s : bias.dim { got %s } != [IC] = [%d]",
                name, Arrays.toString(bias.dim()), IC));
        if(B != null) B.delete();
        B.tensor(bias);
        return this;
    }
    
    public static boolean default_pre_alloc_forward = true;
    protected boolean pre_alloc_forward = default_pre_alloc_forward;
    public boolean pre_alloc_forward() { return this.pre_alloc_forward; }
    public Deconv2D pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append("{ biased = ").append(biased);
        sb.append(", [in_channels, out_channels] = [").append(IC).append(", ").append(OC).append("]");
        sb.append(", kernel  = [").append(FW);
        sb.append(", stride  = [").append(sw).append(']');
        sb.append(", padding = [").append(pw).append(']');
        sb.append(", output_size = [").append(OW);
        sb.append(" }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: others">
    transient private boolean baseW = true;//is the first gradient of W
    transient private boolean baseB = true;//is the first gradient of B
    
    protected final synchronized void reset_backward() { baseW = baseB = true;  }
    protected final synchronized boolean baseW() { boolean old = baseW; baseW = false; return old; }
    protected final synchronized boolean baseB() { boolean old = baseB; baseB = false; return old; }
    
    @Override
    protected InlineDeconv2D create_unit_core() {
        return new InlineDeconv2D(this);
    }
    
    @Override
    protected void __init__(Engine eg) {
        Parameter.delete(W, B);
        
        Tensor tW = eg.empty(IC, FW, OC);
        W = new Parameter(tW).need_grads(true);
        eg.kaiming_uniform(tW.c(), fans(), (float)Math.sqrt(5.0));//inplace: tW
        
        if(biased) {
            float bound = (float) (1.0 / Math.sqrt(FW * IC));
            Tensor tB = eg.uniform(-bound, bound, OC);
            B = new Parameter(tB).need_grads(true);
        }
        
       Parameter.delete(W, B);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: param & state">
    protected String weight_key() { return name + ".weight"; }
    protected String bias_key() { return name + ".bias"; }
    
    @Override
    public void params(ParamSet set) {
        set.add(W);
        if(biased) set.add(B);
    }
    
    @Override
    public void param_map(ParamMap<String> map) {
        map.put(weight_key(), W);
        if(biased) map.put(bias_key(), B);
    }
    
    @Override
    public void state(State dic) {
        dic.put(weight_key(), W.ts());
        if(biased) dic.put(bias_key(), B.ts());
    }
    
   @Override
    public void update_state(State dic, boolean partial) {
        W.ts().set(dic.get(weight_key()), partial, name + ": fail to update state for weight");
        if(biased) B.ts().set(dic.get(bias_key()), partial, name + ": fail to update state for bias");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineDeconv2D">
    public static class InlineDeconv2D extends SimpleCore<Deconv2D> {
        public InlineDeconv2D(Deconv2D unit) { super(unit); }
        
        transient private int IW;
        transient private Tensor y;
        
        //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
        public boolean biased() { return ut.biased; }
        public int out_channel() { return ut.OC; }
        public int in_channel()  { return ut.IC; }
    
        public int[] kernel()   { return new int[]{ ut.FW }; }
        public int[] stride()   { return new int[]{ ut.sw }; }
        public int[] padding()  { return new int[]{ ut.pw }; }
        public int[] out_size() { return new int[]{ ut.OW }; }
     
        public boolean pre_alloc_forward() { return ut.pre_alloc_forward; }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
        @Override
        protected void __before_forward__(Engine eg, Tensor X) { 
            ut.reset_backward();
            IW = X.dim(-2);
            y = (ut.pre_alloc_forward ? //X[N, IW, IC] -> Y[N, OW, OC]
                    eg.alloc.deconv2D(X, ut.W.ts(), ut.OW, ut.sw, ut.pw):
                    null);
        }
        
        @Override//when forward prop: W logically -> Wr
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(ut.pre_alloc_forward) {
                Tensor out = y.c(); y = null;
                return (ut.biased ? //X[N, IW, IC] -> Y[N, OW, OC]
                        eg.deconv2D_biased(out, X, ut.W.ts(), ut.sw, ut.pw, ut.B.ts()) :
                        eg.deconv2D(out, X, ut.W.ts(), ut.sw, ut.pw));
            }
            
            return (ut.biased ? //X[N, IW, IC] -> Y[N, OW, OC]
                    eg.deconv2D_biased(X, ut.W.ts(), ut.OW, ut.sw, ut.pw, ut.B.ts()) :
                    eg.deconv2D(X, ut.W.ts(), ut.OW, ut.sw, ut.pw));
        }
        //</editor-fold>

        //<editor-fold defaultstate="collapsed" desc="running-area: backward-propagation">
        @Override//when backward prop: conv(X, deltaY) -> gradient(Wr)^r -> gradient(W)
        protected Tensor __backward__(Engine eg, Tensor deltaY, 
            boolean grad_inplace, boolean backward_grads) 
        {
            Tensor deltaW = null, deltaB = null, deltaX = null;
            int gc_count = 0;
        
            //conv.deltaY.shape = dconv.X.shape, conv.X.shape = dconv.deltaY.shape
            if(ut.W.need_grads()) {//find the gradient of filters
                deltaW = eg.deconv2D_deltaW(deltaY, holdX(), ut.FW, ut.sw, ut.pw);
                ut.W.accumulate(ut.baseW(), deltaW);
                if(grad_inplace) gc_count++;
            }
        
            if(ut.biased && ut.B.need_grads()) {//find the gradient of bias
                deltaB = eg.field_sum(deltaY, ut.OC);
                ut.B.accumulate(ut.baseB(), deltaB);
                if(grad_inplace) gc_count++;
            }
        
            if(backward_grads) {//find the gradient of input-features
                deltaX = eg.deconv2D_deltaX(deltaY, ut.W.ts(), IW, ut.sw, ut.pw);
                ut.W.ts().follow(deltaX);//When compute deltaX, W can't be changed
                if(grad_inplace) gc_count++;
            }
         
            if(gc_count != 0) {//when deltaW, deltaB, deltaX are cauculated, deltaY is not needed
                CountGc gc = new CountGc(gc_count, deltaY);
                if(deltaW != null) { deltaW.dual(()-> { gc.countDown(); }).remote_sync(); }
                if(deltaB != null) { deltaB.dual(()-> { gc.countDown(); }).remote_sync(); }
                if(deltaX != null) { deltaX.dual(()-> { gc.countDown(); }); }
            }
       
            return deltaX;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
