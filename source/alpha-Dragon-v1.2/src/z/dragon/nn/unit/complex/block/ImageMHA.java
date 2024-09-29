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
 * seq_length = channel, seq_dim = height * width / head
 * @author Gilgamesh
 */
public class ImageMHA extends Module {
    private static final long serialVersionUID = 1L;
    
    public int head;
    public Unit mask;
    
    public ImageMHA(int head, Unit mask) {
        this.head = head;
        this.mask = mask;
    } 
    
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
        
        final int[] Ydim = K.dim();
        final int QN = (Q.ndim() > 2 ? Q.dim(0) : 1);
        final int KN = (K.ndim() > 2 ? K.dim(0) : 1);
        final int VN = (V.ndim() > 2 ? V.dim(0) : 1);
        if (check) {
            if (QN != KN) throw new IllegalArgumentException(String.format("Q.batch { got %d } != K.batch { got %d } ", QN, KN));
            if (QN != VN) throw new IllegalArgumentException(String.format("Q.batch { got %d } != V.batch { got %d } ", QN, VN));
        }
        
        final int QC = Q.dim(-1), KC = K.dim(-1), VC = V.dim(-1);//seq_length = channel
        if (QC != VC) throw new IllegalArgumentException(String.format("Q.seq_len { got %d } != V.seq_len { got %d } ", QC, VC));
        
        final int QL = Q.length() / (QN * QC);//seq_dim = height * width
        final int KL = K.length() / (KN * KC);//seq_dim * seq_len * batch = tensor.length
        final int VL = V.length() / (VN * VC);
        if (QL != KL) throw new IllegalArgumentException(String.format("Q.seq_dim { got %d } != K.seq_dim { got %d }", QL, KL));
        
        final int dKL = KL / head, dVL = VL / head;
        if (check) {
            if (KL % head != 0) throw new IllegalArgumentException(String.format(" K.seq_dim { got %d } mod head { got %d } != 0", KL, head));
            if (VL % head != 0) throw new IllegalArgumentException(String.format(" V.seq_dim { got %d } mod head { got %d } != 0", VL, head));
        }
        
        final int N_head = QN * head;
        Q = F.view(Q, N_head, dKL, VC)[0];
        K = F.view(K, N_head, dKL, KC)[0];
        V = F.view(V, N_head, dVL, VC)[0];
        
        Tensor P = F.batchMatMulT1(Q, K)[0];//[N * head, (VC, dKL)] * [N * head, (dKL, KC] -> [N * head, (VC, KC)]
        P = F.softmax(-1, F.sdiv((float) Math.sqrt(dKL), P))[0];
        if (mask != null) P = mask.forward(P)[0];
            
        Tensor[] Y = F.batchMatMulT2(V, P);//[N * head, (dVL, VC)] * [N * head, (VC, KC)] -> [N * head, (dVL, KC)]
        
        Ydim[Ydim.length - 1] = KC;
        return F.view(Y, Ydim);//[N, head * dVL, KC] ->  [N, H, W, KC]
    }
}
