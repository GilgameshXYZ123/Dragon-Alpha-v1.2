/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

/**
 *
 * @author Gilgamesh
 */
public class EngineAlloc 
{
    protected final Engine eg;
    public EngineAlloc(Engine engine) { this.eg = engine; }
    
    //<editor-fold defaultstate="collapsed" desc="fullconnect">
    public Tensor fullconnect(Tensor X, Tensor W) {
        int ndimX = X.ndim();
        if(eg.check) {
            eg.require_dtype(X, "X"); eg.require_dtype(W, "W");
            eg.must_greater_equal(ndimX, "X.ndim", 2);
            eg.equals(W.ndim(), "W.ndim", 2);
            eg.equals(X.dim(-1), "X.features", W.dim(0), "W.in_features");
        }
       
        int[] Xdim = X.dim, Ydim = new int[ndimX]; 
        for(int i=0; i<ndimX - 1; i++) Ydim[i] = Xdim[i];
        Ydim[ndimX - 1] = W.dim[1];
        return eg.empty(Ydim);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="conv3D / deconv3D">
    public Tensor conv3D(Tensor X, Tensor W, int OH, int OW, int sh, int sw, int ph, int pw) {
        if(eg.check) {
            eg.require_dtype(X); eg.require_dtype(W); 
            eg.equals(X.ndim(), "X.ndim", 4);
            eg.equals(W.ndim(), "W.ndim", 4);
            eg.equals(W.dim(3), "W,IC ", X.dim(3), "X.IC");
        }
        
        int[] dimX = X.dim, dimW = W.dim;
        int XN  = dimX[0], IH = dimX[1], IW = dimX[2];//X[N, IH, IW, IC]
        int WOC = dimW[0], FH = dimW[1], FW = dimW[2];//W[OC, FH, FW, IC]
        
        if(OH == -1) OH = (IH - FH + (ph << 1)) / sh + 1;//floor
        if(OW == -1) OW = (IW - FW + (pw << 1)) / sw + 1;//floor
        return eg.empty(XN, OH, OW, WOC);
    }
    
    public Tensor deconv3D(Tensor X, Tensor W, int OH, int OW, int sh, int sw, int ph, int pw) {
        if(eg.check) {
            eg.require_dtype(X); eg.require_dtype(W); 
            eg.equals(X.ndim(), "X.ndim", 4);
            eg.equals(W.ndim(), "W.ndim", 4);
            eg.equals(W.dim(0), "W.IC", X.dim(3), "X.IC");
        }
        
        int[] dimX = X.dim, dimW = W.dim;
        int XN = dimX[0], IH = dimX[1], IW  = dimX[2];//X[N, IH, IW, IC]
        int FH = dimW[1], FW = dimW[2], WOC = dimW[3];//W[IC, FH, FW, OC]
        
        if(OH == -1) OH = (IH - 1)*sh + FH - (ph << 1);//floor
        if(OW == -1) OW = (IW - 1)*sw + FW - (pw << 1);//floor
        return eg.empty(XN, OH, OW, WOC);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="poo2D">
    public Tensor pool2D(Tensor X, int FH, int FW, int OH, int OW, int sh, int sw, int ph, int pw) {
        if(eg.check) { eg.require_dtype(X, "X"); eg.equals(X.ndim(), "X.ndim", 4); }
        
        int[] Xdim = X.dim;//X[ N, IH, IW, IC]
        int XN = Xdim[0], IH = Xdim[1], IW = Xdim[2], XIC = Xdim[3];
        
        if(OH == -1) OH = (IH - FH + (ph << 1))/sh + 1;//floor
        if(OW == -1) OW = (IW - FW + (pw << 1))/sw + 1;//floor
        return eg.empty(XN, OH, OW, XIC).c();
    }
    
    public Tensor[] pool2D_indexed(Tensor X,
            int FH, int FW, int OH, int OW,
            int sh, int sw, int ph, int pw)
    {
        if(eg.check) { eg.require_dtype(X, "X"); eg.equals(X.ndim(), "X.ndim", 4); }
        
        int[] Xdim = X.dim;//X[ N, IH, IW, IC]
        int XN = Xdim[0], IH = Xdim[1], IW = Xdim[2], XIC = Xdim[3];
        
        if(OH == -1) OH = (IH - FH + (ph << 1))/sh + 1;//floor
        if(OW == -1) OW = (IW - FW + (pw << 1))/sw + 1;//floor
        
        Tensor Y = eg.empty(XN, OH, OW, XIC);
        Tensor Index = eg.empty_int32(XN, OH, OW, XIC);
        return new Tensor[] { Y, Index };
    }
    //</editor-fold>
}
