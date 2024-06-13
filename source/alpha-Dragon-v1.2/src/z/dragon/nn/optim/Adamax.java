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
public class Adamax extends Optimizer
{
    protected float lr_t;
    
    protected float beta1, a1, a2, expBeta1;
    protected Tensor[] V;
     
    protected float beta2, eps;
    protected Tensor[] S;
    
    protected float L1 = 0.0f;
    protected float L2 = 0.0f;
    
    //<editor-fold defaultstate="collapsed" desc="__init__">
    private void __init__(float beta1, float beta2, float eps) {
        this.beta1 = beta1; 
        this.beta2 = beta2; this.eps = eps;
        
        V = Tensor.zero_like(params); expBeta1 = 1.0f; 
        S = Tensor.zero_like(params);
        Tensor.sync(V); Tensor.sync(S);
    } 
    //</editor-fold>
    public Adamax(Parameter[] params, float lr, float beta1, float beta2, float eps) {
        super(params, lr);
        __init__(beta1, beta2, eps);
    }
    
    public Adamax(Collection<Parameter> params, float lr, float beta1, float beta2, float eps) {
        super(params, lr);
        __init__(beta1, beta2, eps);
    }
    
    public Adamax(Map<String, Parameter> param_map, float lr, float beta1, float beta2, float eps) {
        super(param_map, lr);
        __init__(beta1, beta2, eps);
    }
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
    @Override public Adamax learning_rate(float lr) { super.learning_rate(lr); return this; }
    
    public Tensor[] exp_avg() { return V; }
    public Tensor[] exp_avg_sq() { return S; }
    
    public float beta1() {return beta1;}
    public Adamax beta1(float beta1) { this.beta1 = beta1;  return this; }
    
    public float beta2() { return beta2; }
    public Adamax beta2(float beta2) { this.beta2 = beta2; return this; }
    
    public float eps() { return eps; }
    public Adamax eps(float eps) { this.eps = eps; return this;}
    
    public float L1() { return L1; }
    public Adamax L1(float L1) { this.L1 = L1; return this; }
    
    public float L2() {return L2;}
    public Adamax L2(float L2) { this.L2 = L2; return this; }
    
    @Override
    public void append(StringBuilder sb) {
        sb.append(getClass().getSimpleName());
        sb.append(" { learning_rate = ").append(lr);
        sb.append(", [beta1, beta2, eps] = [")
                .append(beta1).append(", ")
                .append(beta2).append(", ")
                .append(eps).append("]");
        sb.append(", [L1, L2] = [")
                .append(L1).append(", ")
                .append(L2).append("] }");
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: state">
    //hypher-state--------------------------------------------------------------
    @Override
    protected void hypher_state(State dic) {
        dic.put("expBetas", State.floats(expBeta1));
    }
    
    @Override
    protected void update_hypher_state(State dic, boolean partial) {
        StateValue expBetas = dic.get("expBetas");
        State.set(expBetas, "fail to load expBetas", partial, ()->{
            float[] arr = Vector.to_float_vector(expBetas.toStringLines(), 2);
            expBeta1 = arr[0];
        });
    }

    //param-state---------------------------------------------------------------
    protected String exp_avg_key(String param_name) { return param_name + ".exp_avg"; }
    protected String exp_avg_sq_key(String param_name) { return param_name + ".exp_avg_sq"; }
    
    @Override
    protected void param_state(State dic, int index, String paramName) {
        dic.put(exp_avg_key(paramName), V[index]);
        dic.put(exp_avg_sq_key(paramName), S[index]);
    }
    
    @Override
    protected void update_param_state(State dic, boolean partial, int index, String paramName) {
        String exp_avg_key = exp_avg_key(paramName);
        V[index].set(dic.get(exp_avg_key), partial, "fail to load " + exp_avg_key);
        
        String exp_avg_sq_key = exp_avg_sq_key(paramName);
        S[index].set(dic.get(exp_avg_sq_key), partial, "fail to load " + exp_avg_sq_key);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="running-area: update">
    @Override
    protected void __before_update__() {//init: Us = Uv = 1
        expBeta1 *= beta1;
        a1 = beta1; a2 = 1.0f - beta1; //exp_avg_mean
        lr_t = lr / (1.0f - expBeta1);
    }
    
    @Override
    protected void __update__(int index, Tensor grad, Engine eg) {
        Tensor w = params[index].ts();//weight, inplace: param.datas
        Tensor v = V[index];//exp_avg
        Tensor s = S[index];//exp_avg_sq
        boolean decay = (L1 != 0.0f) || (L2 != 0.0f);
        
        if(decay) eg.adamax(w, v, a1, a2, s, beta2, eps, grad, lr_t, L1, L2);
        else eg.adamax(w, v, a1, a2, s, beta2, eps, grad, lr_t);
    }
    
    @Override
    protected void __update__(int index, Collection<Tensor> grads, Engine eg) {
        Tensor w = params[index].ts();//weight, inplace: param.datas
        Tensor v = V[index];//exp_avg
        Tensor s = S[index];//exp_avg_sq
        boolean decay = (L1 != 0.0f) || (L2 != 0.0f);
        
        if(decay) eg.adamax(w, v, a1, a2, s, beta2, eps, grads, lr_t, L1, L2);
        else eg.adamax(w, v, a1, a2, s, beta2, eps, grads, lr_t);
    }
    
    @Override
    protected void __clear__() {
        if(V != null) { Tensor.delete(V); V = null; }
        if(S != null) { Tensor.delete(S); S = null; }
    }
    //</editor-fold>
}
