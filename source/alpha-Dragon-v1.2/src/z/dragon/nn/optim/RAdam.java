/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.optim;

import java.util.Collection;
import java.util.Map;
import z.dragon.common.state.State;
import z.dragon.common.state.State.StateValue;
import z.dragon.engine.Engine;
import z.dragon.engine.Parameter;
import z.dragon.engine.Tensor;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class RAdam extends Optimizer {
    protected float lr_t, eps_t;
    protected int t;//step
    
    protected float beta1, a1, a2, expBeta1;
    protected Tensor[] V;
     
    protected float beta2, eps, b1, b2, expBeta2;
    protected Tensor[] S;
    
    protected float pcur, pinf;
    
    protected float L1 = 0.0f;
    protected float L2 = 0.0f;

    //<editor-fold defaultstate="collapsed" desc="__init__">
    private void __init__(float beta1, float beta2, float eps) {
        this.beta1 = beta1; t = 0;
        this.beta2 = beta2; this.eps = eps;
        
        V = Tensor.zero_like(params); expBeta1 = 1.0f; 
        S = Tensor.zero_like(params); expBeta2 = 1.0f;
        Tensor.sync(V); Tensor.sync(S);
    } 
    //</editor-fold>
    public RAdam(Parameter[] params, float lr, float beta1, float beta2, float eps) {
        super(params, lr);
        __init__(beta1, beta2, eps);
    }
    
    public RAdam(Collection<Parameter> params, float lr, float beta1, float beta2, float eps) {
        super(params, lr);
        __init__(beta1, beta2, eps);
    }
    
    public RAdam(Map<String, Parameter> params, float lr, float beta1, float beta2, float eps) {
        super(params, lr);
        __init__(beta1, beta2, eps);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public RAdam learning_rate(float lr) { super.learning_rate(lr); return this; }
    
    public Tensor[] exp_avg() { return V; }
    public Tensor[] exp_avg_sq() { return S; }
    
    public int step() { return t; }
    
    public float beta1() {return beta1;}
    public RAdam beta1(float beta1) { this.beta1 = beta1;  return this; }
    
    public float beta2() { return beta2; }
    public RAdam beta2(float beta2) { this.beta2 = beta2; return this; }
    
    public float eps() { return eps; }
    public RAdam eps(float eps) { this.eps = eps; return this;}
    
    public float L1() { return L1; }
    public RAdam L1(float L1) { this.L1 = L1; return this; }
    
    public float L2() { return L2; }
    public RAdam L2(float L2) { this.L2 = L2; return this; }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append(" { learning_rate = ").append(lr);
        sb.append(", [beta1, beta2, eps] = [")
                .append(beta1).append(", ")
                .append(beta2).append(", ")
                .append(eps).append("]");
        sb.append(", [L1, L2] = (")
                .append(L1).append(", ")
                .append(L2).append(") }");
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: state">
    //hypher-state--------------------------------------------------------------
    @Override
    protected void hypher_state(State dic) {
        dic.put("expBetas", State.floats(expBeta1, expBeta2));
        dic.put("step", State.ints(t));
    }
    
    @Override
    protected void update_hypher_state(State dic, boolean partial) {
        StateValue expBetas = dic.get("expBetas");
        State.set(expBetas, "fail to load expBetas", partial, ()->{
            float[] arr = Vector.to_float_vector(expBetas.toStringLines(), 2);
            expBeta1 = arr[0]; expBeta2 = arr[1];
        });
        
        StateValue step = dic.get("step");
        State.set(step, "fail to load step", partial, ()->{
            t = Vector.to_int_vector(step.toStringLines(), 1)[0];
        });
    }

    //param-state---------------------------------------------------------------
    protected String exp_avg_key(String param_name) { return param_name + ".exp_avg"; }
    protected String exp_avg_sq_key(String param_name) { return param_name + ".exp_avg_sq"; }
    
    @Override
    protected void param_state(State dic, int index, String param_name) {
        dic.put(exp_avg_key(param_name), V[index]);
        dic.put(exp_avg_sq_key(param_name), S[index]);
    }
    
    @Override
    protected void update_param_state(State dic, boolean partial, int index, String param_name) {
        String exp_avg_key = exp_avg_key(param_name);
        V[index].set(dic.get(exp_avg_key), partial, "fail to load " + exp_avg_key);
        
        String exp_avg_sq_key = exp_avg_sq_key(param_name);
        S[index].set(dic.get(exp_avg_sq_key), partial, "fail to load " + exp_avg_sq_key);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="running-area: update">
    @Override
    protected void __before_update__() {
        t++; expBeta1 *= beta1; expBeta2 *= beta2; //step += 1
        
        a1 = beta1; a2 = 1.0f - beta1;//exp_avg
        b1 = beta2; b2 = 1.0f - beta2;//exp_avg_sq
         
        pinf = 2.0f / (1.0f - beta2) - 1.0f;
        pcur = pinf - 2.0f * t * expBeta2 / (1.0f - expBeta2);
      
        lr_t = lr;
        if(pcur > 5.0f) {//lr_t = lr * r * correct_beta2/correct_beta1 * V/sqrt(S + eps)
            double ra = (pcur - 4.0) * (pcur - 2.0) * pinf;
            double rb = (pinf - 4.0) * (pinf - 2.0) * pcur;
            float r = (float) Math.sqrt(ra / rb);
            
            double correct_beta2 = Math.sqrt(1 - expBeta2);
            lr_t  = (float) (lr_t * r * correct_beta2);
            eps_t = (float) (eps * correct_beta2);
        } 
        lr_t = lr_t / (1.0f - expBeta1);
    }

    @Override
    protected void __update__(int index, Tensor grad, Engine eg) {
        Tensor w = params[index].ts();//weight, inplace: param.datas
        Tensor v = V[index];//exp_avg
        Tensor s = S[index];//exp_avg_sq
        boolean decay = (L1 != 0.0f) || (L2 != 0.0f);
        
        if(decay) eg.radam(w, v, a1, a2, s, b1, b2, eps_t, grad, pcur, lr_t, L1, L2);
        else eg.radam(w, v, a1, a2, s, b1, b2, eps_t, grad, pcur, lr_t); 
    }

    @Override
    protected void __update__(int index, Collection<Tensor> grads, Engine eg) {
        Tensor w = params[index].ts();//weight, inplace: param.datas
        Tensor v = V[index];//exp_avg
        Tensor s = S[index];//exp_avg_sq
        boolean decay = (L1 != 0.0f) || (L2 != 0.0f);
        
        if(decay) eg.radam(w, v, a1, a2, s, b1, b2, eps_t, grads, pcur, lr_t, L1, L2);
        else eg.radam(w, v, a1, a2, s, b1, b2, eps_t, grads, pcur, lr_t); 
    }

    @Override
    protected void __clear__() {
        if(V != null) { Tensor.delete(V); V = null; }
        if(S != null) { Tensor.delete(S); S = null; }
    }
    //</editor-fold>
}
