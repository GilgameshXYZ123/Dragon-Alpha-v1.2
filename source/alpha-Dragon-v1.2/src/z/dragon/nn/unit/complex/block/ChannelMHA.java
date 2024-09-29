/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.nn.unit.complex.block;

import static z.dragon.alpha.Alpha.UnitFunctional.F;
import z.dragon.engine.Tensor;
import z.dragon.nn.unit.Unit;
import z.dragon.nn.unit.complex.Module;

/**
 * seq_length = height * width, seq_dim = channel / head
 * @author Gilgamesh
 */
public class ChannelMHA extends Module{
    private static final long serialVersionUID = 1L;
    
    public int head;
    public Unit mask;
    
    public ChannelMHA(int head, Unit mask) {
        this.head = head;
        this.mask = mask;
    }
    
    public int head() { return head; }
    public Unit mask() { return mask; }

    @Override
    public Tensor[] __forward__(Tensor... X) {
        if (X.length != 3) throw new IllegalArgumentException(String.format("X.length { got %d } != 3", X.length));
        Tensor Q = X[0], K = X[1], V = X[2];//[(N, H, W), C]
        boolean check = Q.engine().check();
        if (check) {
            if (Q.ndim() < 2) throw new IllegalArgumentException(String.format("Q.ndim { got %d } < 2", Q.ndim()));
            if (K.ndim() < 2) throw new IllegalArgumentException(String.format("K.ndim { got %d } < 2", K.ndim()));
            if (V.ndim() < 2) throw new IllegalArgumentException(String.format("V.ndim { got %d } < 2", V.ndim()));
        }
        
        final int[] Ydim = Q.dim();
        final int QN  = (Q.ndim() > 2 ? Q.dim(0) : 1);
        final int KN =  (K.ndim() > 2 ? K.dim(0) : 1);
        final int VN =  (V.ndim() > 2 ? V.dim(0) : 1);
        if (check) {
            if (QN != KN) throw new IllegalArgumentException(String.format("Q.batch { got %d } != K.batch { got %d } ", QN, KN));
            if (QN != VN) throw new IllegalArgumentException(String.format("Q.batch { got %d } != V.batch { got %d } ", QN, VN));
        }
        
        final int QC = Q.dim(-1), KC = K.dim(-1), VC = V.dim(-1);//seq_dim = channel
        if (QC != KC) throw new IllegalArgumentException(String.format("Q.seq_dim { got %d } != K.seq_dim { got %d } ", QC, KC));
        
        final int QL = Q.length() / (QN * QC);//seq_dim = height * width
        final int KL = K.length() / (KN * KC);//seq_dim * seq_len * batch = tensor.length
        final int VL = V.length() / (VN * VC);
        if (KL != VL) throw new IllegalArgumentException(String.format("K.seq_len { got %d } != V.seq_len { got %d }", KL, VL));
        
        final int dKC = KC / head, dVC = VC / head;
        if (check) {
            if (dKC % head != 0) throw new IllegalArgumentException(String.format(" K.seq_dim { got %d } mod head { got %d } != 0", dKC, head));
            if (dVC % head != 0) throw new IllegalArgumentException(String.format(" V.seq_dim { got %d } mod head { got %d } != 0", dVC, head));
        }
        
        final int N = QN;
        Q = F.view(Q, N, QL, head, dKC)[0];
        K = F.view(K, N, VL, head, dKC)[0];
        V = F.view(V, N, VL, head, dVC)[0];
            
        Q = F.transpose(1, 2, Q)[0];//[N, head, QL, dKC]
        K = F.transpose(1, 2, K)[0];//[N, head, VL, dKC]
        V = F.transpose(1, 2, V)[0];//[N, head, VL, dVC]
        
        Tensor P = F.batchMatMulT2(Q, K)[0];//[N, head, (QL, dKC)] * [N, head, (dKC, VL)] -> [N, head, (QL, VL)]
        P = F.softmax(-1, F.sdiv((float) Math.sqrt(KC), P))[0];//[N, head, L, L]
        if (mask != null) P = mask.forward(P)[0];
            
        Tensor Y = F.batchMatMul(P, V)[0];//[N, head, (QL, VL)] * [N, head,(VL, dVC)] = [N, head, (QL, dVC)]
        
        Y = F.transpose(1, 2, Y)[0];//[N, head, QL, dVC] -> [N, QL, head, dVC]
        Ydim[Y.length() - 1] = VC;
        return F.view(Y, Ydim);//[N, QL, head * dVC] -> [N, H, W, VC]
    }
}
