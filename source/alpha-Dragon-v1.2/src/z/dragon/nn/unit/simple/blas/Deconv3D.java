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
public class Deconv3D extends SimpleUnit {
    private static final long serialVersionUID = 562781240120001L;
    
    protected int IC, OC;
    protected int FH, FW;
    protected int sh, sw;
    protected int ph, pw;
    protected int OH, OW;
    protected boolean biased;
    
    transient protected Parameter W;
    transient protected Parameter B;
    
    public Deconv3D(boolean biasd,
            int in_channel,     int out_channel, 
            int kernel_height,  int kernel_width, 
            int stride_height,  int stride_width,
            int padding_height, int padding_width,
            int output_height,  int output_width) 
    {
        this.biased = biasd;
        this.IC = in_channel;     this.OC = out_channel;
        this.FH = kernel_height;  this.FW = kernel_width;
        this.sh = stride_height;  this.sw = stride_width;
        this.ph = padding_height; this.pw = padding_width;
        this.OH = output_height;  this.OW = output_width;
    }

    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    public boolean biased() { return biased; }
    public int out_channel() { return OC; }
    public int in_channel()  { return IC; }
    
    public int[] kernel()  { return new int[]{ FH, FW }; }
    public int[] stride()  { return new int[]{ sh, sw }; }
    public int[] padding() { return new int[]{ ph, pw }; }
    public int[] fans() { return new int[]{ FH*FW*OC, FH*FW*IC }; }//[fan_in, fan_out]
    
    public final int[] out_size() { return new int[] { OH, OW }; }
    public Deconv3D out_size(int... out_size) {
        if(out_size == null || out_size.length != 2) throw new IllegalArgumentException();
        OH = out_size[0]; OW = out_size[1];
        return this;
    }
    
    public Parameter weight_param() { return W; }
    public Tensor weight() { return W.ts(); }
    public Deconv3D weight(Tensor weight) {
        if(Tensor.isNull(weight)) throw new NullPointerException("weight is null");
        if(!weight.dimEquals(IC, FH, FW, OC)) throw new IllegalArgumentException(String.format(
                "%s : weight.dim { got %s } != [IC, FH, FW, OC] = [%d, %d, %d, %d]", 
                name, Arrays.toString(weight.dim()), IC, FH, FW, OC));
        if(W != null) W.delete(); 
        W.tensor(weight);
        return this;
    }
    
    public Parameter bias_param() { return B; }
    public Tensor bias() { return B.ts(); }
    public Deconv3D bias(Tensor bias) {
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
    public Deconv3D pre_alloc_forward(boolean flag) { pre_alloc_forward = flag; return this; }
    
    @Override
    public void append(String pre, StringBuilder sb) {
        sb.append(pre).append(default_name());
        sb.append("{ biased = ").append(biased);
        sb.append(", [in_channels, out_channels] = [").append(IC).append(", ").append(OC).append("]");
        sb.append(", kernel  = [").append(FH).append(", ").append(FW);
        sb.append(", stride  = [").append(sh).append(", ").append(sw).append(']');
        sb.append(", padding = [").append(ph).append(", ").append(pw).append(']');
        sb.append(", output_size = [").append(OH).append(OW);
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
    protected InlineDeconv3D create_unit_core() {
        return new InlineDeconv3D(this);
    }
    
    @Override
    protected void __init__(Engine eg) {
        Parameter.delete(W, B);
        
        Tensor tW = eg.empty(IC, FH, FW, OC);
        W = new Parameter(tW).need_grads(true);
        eg.kaiming_uniform(tW.c(), fans(), (float)Math.sqrt(5.0));//inplace: tW
        
        if(biased) {
            float bound = (float) (1.0 / Math.sqrt(FH * FW * IC));
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
    
    //<editor-fold defaultstate="collapsed" desc="static class: InlineDeconv3D">
    public static class InlineDeconv3D extends SimpleCore<Deconv3D> {
        public InlineDeconv3D(Deconv3D unit) { super(unit); }
        
        transient private int IH, IW;
        transient private Tensor y;
        
        //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
        public boolean biased() { return ut.biased; }
        public int out_channel() { return ut.OC; }
        public int in_channel()  { return ut.IC; }
    
        public int[] kernel()   { return new int[]{ ut.FH, ut.FW }; }
        public int[] stride()   { return new int[]{ ut.sh, ut.sw }; }
        public int[] padding()  { return new int[]{ ut.ph, ut.pw }; }
        public int[] out_size() { return new int[]{ ut.OH, ut.OW }; }
     
        public boolean pre_alloc_forward() { return ut.pre_alloc_forward; }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="running-area: forward-propagation">
        @Override
        protected void __before_forward__(Engine eg, Tensor X) { 
            ut.reset_backward();
            IH = X.dim(-3); IW = X.dim(-2);
            y = (ut.pre_alloc_forward ? //X[N, IH, IW, IC] -> Y[N, OH, OW, OC]
                    eg.alloc.deconv3D(X, ut.W.ts(), ut.OH, ut.OW, ut.sh, ut.sw, ut.ph, ut.pw):
                    null);
        }
        
        @Override//when forward prop: W logically -> Wr
        protected Tensor __forward__(Engine eg, Tensor X) {
            if(ut.pre_alloc_forward) {
                Tensor out = y.c(); y = null;
                return (ut.biased ? //X[N, IH, IW, IC] -> Y[N, OH, OW, OC]
                        eg.deconv3D_biased(out, X, ut.W.ts(), ut.sh, ut.sw, ut.ph, ut.pw, ut.B.ts()) :
                        eg.deconv3D(out, X, ut.W.ts(), ut.sh, ut.sw, ut.ph, ut.pw));
            }
            
            return (ut.biased ? //X[N, IH, IW, IC] -> Y[N, OH, OW, OC]
                    eg.deconv3D_biased(X, ut.W.ts(), ut.OH, ut.OW, ut.sh, ut.sw, ut.ph, ut.pw, ut.B.ts()) :
                    eg.deconv3D(X, ut.W.ts(), ut.OH, ut.OW, ut.sh, ut.sw, ut.ph, ut.pw));
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
                deltaW = eg.deconv3D_deltaW(deltaY, holdX(), ut.FH, ut.FW, ut.sh, ut.sw, ut.ph, ut.pw);
                ut.W.accumulate(ut.baseW(), deltaW);
                if(grad_inplace) gc_count++;
            }
        
            if(ut.biased && ut.B.need_grads()) {//find the gradient of bias
                deltaB = eg.field_sum(deltaY, ut.OC);
                ut.B.accumulate(ut.baseB(), deltaB);
                if(grad_inplace) gc_count++;
            }
        
            if(backward_grads) {//find the gradient of input-features
                deltaX = eg.deconv3D_deltaX(deltaY, ut.W.ts(), IH, IW, ut.sh, ut.sw, ut.ph, ut.pw);
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
