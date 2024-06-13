/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.engine;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.charset.Charset;
import java.util.Arrays;
import z.dragon.common.DragonCV;
import z.dragon.common.DragonCV.DimPixel;
import static z.dragon.common.DragonCV.cv;
import static z.dragon.common.DragonFile.fl;
import static z.dragon.engine.Engine.from_center;
import z.dragon.engine.Syncer.ChainSyncer;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class ImageEngine 
{
    protected final Engine eg;
    protected ImageEngine(Engine engine) { this.eg = engine; }
    
    //<editor-fold defaultstate="collapsed" desc="image: Tensor: valueOf">
    public byte[] pixel(Tensor X) { return eg.valueOf_int8(X); }
    public byte[][] pixel_channel_split(Tensor X) {
        eg.must_greater_equal(X.ndim(), "X.ndim", 3);
        return cv.pixel_channel_split(pixel(X), X.dim());
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: Tensor: create">
    public Tensor tensor(byte[][] pixels, int... dim) { return tensor(1.0f, pixels, 0.0f, dim); }
    public Tensor tensor(float alpha, byte[][] pixels, float beta, int... dim) {
        if(eg.check) { eg.checkMatrix(pixels, "pixels"); }
        byte[] pixel = eg.toByteVector(pixels);
        return tensor(alpha, pixel, beta, dim);
    }
    
    public Tensor tensor(byte[] pixel, int...dim) { return tensor(1.0f, pixel, 0.0f, dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor tensor(float alpha, byte[] pixel, float beta, int...dim) {
        dim = eg.negative_dim(pixel.length, dim);
        Tensor BX = eg.tensor_int8(pixel, dim);
        Tensor Y = eg.empty(dim).c();
        Syncer sc = eg.core.linear2D_pixel_to_dtype(Y.address, 
                alpha, BX.address, beta, 
                Y.lengthv, Y.lastDim());
        sc.sync(); eg.delete(BX);
        return Y;
    }
    
    public Tensor pixels(byte[] pixel, int... dim) { return eg.tensor_int8(pixel, dim); }
    public Tensor pixels(byte[][] pixel, int... dim) { return eg.tensor_int8(pixel, dim); }
    
    public Tensor blacks(int... dim) { return eg.zeros_int8(dim); }
    public Tensor blacks_like(Tensor X) { return eg.zeros_int8(X.dim); }
    
    public Tensor whites(int... dim) { return eg.constants_int8(255, dim);}
    public Tensor whites_like(Tensor X) { return eg.constants_int8(255, X.dim);}
    
    public Tensor constants_like(int value, Tensor X) {
        if(value > 255) value = 255; else if(value < 0) value = 0;
        return eg.constants_int8(value, X.dim);
    }
    public Tensor constants(int value, int... dim) {
        if(value > 255) value = 255; else if(value < 0) value = 0;
        return eg.constants_int8(value, dim);
    }
    
    public Tensor pixels(BufferedImage img) { return eg.tensor_int8(cv.pixel(img), cv.dim(img)); }
    public Tensor pixels(BufferedImage... imgs) {
        if(eg.check) {
            if(imgs == null || imgs.length == 0) throw new NullPointerException("imgs == null || imgs.length == 0");
            for(int i=1, dim0[] = cv.dim(imgs[0]); i<imgs.length; i++) 
                if(!Arrays.equals(dim0, cv.dim(imgs[i]))) throw new IllegalArgumentException(String.format(
                        "All images must have the same dimensions [%s]", Arrays.toString(dim0)));
        }
        
        int N = imgs.length;
        int[] dim = new int[] { N,//batch_size 
            imgs[0].getHeight(), imgs[0].getWidth(),//[H, W]
            imgs[0].getColorModel().getNumComponents()//channels
        };
      
        byte[][] pixels = new byte[N][];
        for(int i=0; i<imgs.length; i++) pixels[i] = DragonCV.cv.pixel(imgs[i]);
        byte[] pixel = eg.toByteVector(pixels);
        
        return eg.tensor_int8(pixel, dim);
    }
    
    public Tensor read_raw_bil_dtype12(byte[] bytes, int... dim) { return ImageEngine.this.read_raw_bil_dtype12(bytes, dim[0], dim[1], dim[2]); }
    public Tensor read_raw_bil_dtype12(byte[] bytes, int IH, int IW, int IC) {
        byte[] pixels = cv.read_raw_bil_dtype12(bytes);
        Tensor X = eg.tensor_int8(pixels, IW*IC, IH).c();//[IW, IC, IH]
        X = this.transpose(true, X, 0, 1);//[IH, IW, IC]
        X.setDim(false, IH, IW, IC);
        return X;
    }
    
    public Tensor read_raw_bil_dtype12(String path, int... dim) { return ImageEngine.this.read_raw_bil_dtype12(path, dim[0], dim[1], dim[2]); }
    public Tensor read_raw_bil_dtype12(String path, int IH, int IW, int IC) {
        byte[] pixels = cv.read_raw_bil_dtype12(path);
        Tensor X = eg.tensor_int8(pixels, IW*IC, IH).c();//[(IW, IC), IH]
        X = this.transpose(true, X, 0, 1);//[IH, IW, IC]
        X.setDim(false, IH, IW, IC);
        return X;
    }
    
    public Tensor read_raw_bil_dtype12(File file, int... dim) { return read_raw_bil_dtype12(file, dim[0], dim[1], dim[2]); }
    public Tensor read_raw_bil_dtype12(File file, int IH, int IW, int IC)  {
        byte[] pixels = cv.read_raw_bil_dtype12(file);
        Tensor X = eg.tensor_int8(pixels, IW*IC, IH).c();//[(IW, IC), IH]
        X = this.transpose(true, X, 0, 1);//[IH, (IW, IC)]
        return X.view(true, IH, IW, IC);
    }
    
    public Tensor read_pixels(String path, int... dim) { return read_pixels(path, dim[0], dim[1], dim[2]); }
    public Tensor read_pixels(String path, int IH, int IW, int IC) { return pixels(fl.to_bytes(path), IH, IW, IC); }
    public Tensor read_pixels(File file, int... dim) { return read_pixels(file, dim[0], dim[1], dim[2]); }
    public Tensor read_pixels(File file, int IH, int IW, int IC) { return pixels(fl.to_bytes(file), IH, IW, IC); }
    
    public void write_pixels(String path, Tensor X) { fl.wt_bytes(path, eg.valueOf_int8(X)); }
    public void write_pixels(File file, Tensor X) { fl.wt_bytes(file, eg.valueOf_int8(X)); }
    
    public void write_zip_pixels(String path, Tensor X) { cv.write_zip_pixels(path, eg.valueOf_int8(X), X.dim()); }
    public void write_zip_pixels(File file, Tensor X) { cv.write_zip_pixels(file, eg.valueOf_int8(X), X.dim()); }
    public void write_zip_pixels(File file, Tensor X, int zip_level, Charset charset) { 
        cv.write_zip_pixels(file, zip_level, charset, eg.valueOf_int8(X), X.dim());
    }
    
    public Tensor read_zip_pixels(String path) { DimPixel dp = cv.read_zip_pixels(path); return pixels(dp.pixel, dp.dim); }
    public Tensor read_zip_pixels(File file) { DimPixel dp = cv.read_zip_pixels(file); return pixels(dp.pixel, dp.dim); }
    public Tensor read_zip_pixels(File file, Charset charset) {
        DimPixel dp = cv.read_zip_pixels(file, charset);
        return pixels(dp.pixel, dp.dim);
    }
    //</editor-fold>

    //<editor-fold defaultstate="collapsed" desc="image: pixel(uint8) to dtype">
    //<editor-fold defaultstate="collapsed" desc="linear: pixel to dtype">
    public Tensor pixel_to_dtype(boolean inplace, Tensor X) { return linear_pixel_to_dtype(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_pixel_to_dtype(boolean inplace, float alpha, Tensor X, float beta) {
        if(eg.check) { eg.require_int8(X, "X"); }
        
        Tensor Y = eg.empty(X.dim).c();
        Syncer sc1 = eg.core.linear2D_pixel_to_dtype(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); Y.setSyncer(sc1); return Y;  }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        X.dataType = Y.dataType;//X<int8> -> X<dtype>
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    
    public Tensor dtype_to_pixel(boolean inplace, Tensor X) { return linear_dtype_to_pixel(inplace, 1.0f, X, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor linear_dtype_to_pixel(boolean inplace, float alpha, Tensor X, float beta) {
        if(eg.check) { eg.require_dtype(X, "X"); }
        
        Tensor Y = eg.empty_int8(X.dim).c();
        Syncer sc1 = eg.core.linear2D_dtype_to_pixel(Y.address,
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        X.dataType = Y.dataType;//X<dtype> -> X<int8>
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: dualLinear2_div">
    public Tensor reflection_normalization(boolean inplace, Tensor X, Tensor white, Tensor black) {
        return dualLinear2_div(inplace, black, X, white, 
                -1.0f, 1.0f, 0.0f,//Y1 = -black + X     = X - black
                -1.0f, 1.0f, 0.0f,//Y2 = -black + white = white - black
                0.0f);//Y = Y1 / Y2
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor dualLinear2_div(boolean inplace,
            Tensor X, Tensor X1, Tensor X2,
            float alpha1, float beta1, float gamma1,
            float alpha2, float beta2, float gamma2, float C)
    {
        if(eg.check) {//Y = (alpha1*X + beta1*X1 + gamma1) / (alpha2*X + beta2*X2 + gamma2) + C
            eg.require_int8(X, "X"); eg.require_int8(X1, "X1"); eg.require_int8(X2, "X2");
            eg.equals_valueStructure(X, "X", X1, "X1");
            eg.equals_valueStructure(X, "X", X2, "X2");
        }
        
        Tensor Y = eg.empty(X.dim).c();
        Syncer sc1 = eg.core.img_dualLinear2_div2D(Y.address, 
                X.address, X1.address, X2.address,
                alpha1, beta1, gamma1, 
                alpha2, beta2, gamma2, C,
                X.lengthv, X.lastDim());
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); Y.setSyncer(sc1); return Y;  }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        X.dataType = Y.dataType;//X<int8> -> X<dtype>
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: linear2_div_row">
    public Tensor normalize_row(boolean inplace, Tensor X, Tensor X_mean, Tensor X_std, float eps) {
        return linear2_div_row(inplace, X, X_mean, X_std,
                1.0f, -1.0f, 0.0f,//X - mean
                1.0f, eps, 0.0f);//std + eps
    }
    public Tensor add_div_row(boolean inplace, Tensor X, Tensor X1, Tensor X2) {
        return linear2_div_row(inplace, X, X1, X2,
                1.0f, 1.0f, 0.0f,//X + X1
                1.0f, 0.0f, 0.0f);//X2
    }
    public Tensor sub_div_row(boolean inplace, Tensor X, Tensor X1, Tensor X2) {
        return linear2_div_row(inplace, X, X1, X2,
                1.0f, -1.0f, 0.0f,//X - X1
                1.0f, 0.0f, 0.0f);//X2
    }
    
    @Passed("CudaFloat32EngieBase") //(alpha*X + beta*X1 + gamma) / (alpha2*X2 + beta2) + C
    public Tensor linear2_div_row(boolean inplace, Tensor X, Tensor X1, Tensor X2,
            float alpha1, float beta1, float gamma,
            float alpha2, float beta2, float C)
    {
        if(eg.check) {
            eg.require_int8(X, "X"); eg.require_dtype(X1, "X1"); eg.require_dtype(X2, "X2");
            eg.check_row(X, "X", X1, "X1");
            eg.check_row(X, "X", X2, "X2");
        }
        
        Tensor Y = eg.empty(X.dim).c();
        Syncer sc1 = eg.core.img_linear2_div2D_row(Y.address,
                X.address,
                X1.address, X2.address, X1.lengthv,
                alpha1, beta1, gamma, 
                alpha2, beta2, C, 
                X.lengthv, X.lastDim());
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); Y.setSyncer(sc1); return Y;  }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        X.dataType = Y.dataType;//X<int8> -> X<dtype>
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="image: linear2_div_field">
    public Tensor normalize_field(boolean inplace, Tensor X, Tensor X_mean, Tensor X_std, float eps) {
        return linear2_div_field(inplace, X, X_mean, X_std,
                1.0f, -1.0f, 0.0f,//X - var
                1.0f, eps, 0.0f);//std + eps
    }
    public Tensor sub_div_field(boolean inplace, Tensor X, Tensor X1, Tensor X2) {
        return linear2_div_field(inplace, X, X1, X2,
                1.0f, -1.0f, 0.0f,//X - X1
                1.0f, 0.0f, 0.0f);//X2
    }
    public Tensor add_div_field(boolean inplace, Tensor X, Tensor X1, Tensor X2) {
        return linear2_div_field(inplace, X, X1, X2,
                1.0f, 1.0f, 0.0f,//X + X1
                1.0f, 0.0f, 0.0f);//X2
    }
    
    @Passed("CudaFloat32EngieBase") //(alpha*X + beta*X1 + gamma) / (alpha2*X2 + beta2) + C
    public Tensor linear2_div_field(boolean inplace, Tensor X, Tensor X1, Tensor X2,
            float alpha1, float beta1, float gamma,
            float alpha2, float beta2, float C)
    {
        if(eg.check) {
            eg.require_int8(X, "X"); eg.require_dtype(X1, "X1"); eg.require_dtype(X2, "X2");
            eg.check_field(X, "X", X1, "X1");
            eg.check_field(X, "X", X2, "X2");
        }
        
        Tensor Y = eg.empty(X.dim).c();
        Syncer sc1 = eg.core.img_linear2_div2D_field(Y.address,
                X.address, 
                X1.address, X2.address, X1.length, 
                alpha1, beta1, gamma,
                alpha2, beta2, C, 
                X.lengthv, X.lastDim());
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); Y.setSyncer(sc1); return Y;  }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        X.dataType = Y.dataType;//X<int8> -> X<dtype>
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: elementwise functions">
    //<editor-fold defaultstate="collapsed" desc="image: constant">
    public Tensor white(Tensor X) { return eg.zero_int8(X); }
    public Tensor black(Tensor X) { return eg.constant_int8(X, 255); }
    public Tensor constant(Tensor X, int value) {
        if(value > 255) value = 255; else if(value < 0) value = 0;
        return eg.constant_int8(X, value);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: linear">
    public Tensor sadd(boolean inplace, Tensor X, float C) { return linear(inplace, 1.0f, X, C); }
    public Tensor ssub(boolean inplace, Tensor X, float C) { return linear(inplace, 1.0f, X, -C); }
    public Tensor smul(boolean inplace, Tensor X, float C) { return linear(inplace, C, X, 0.0f); }
    public Tensor sdiv(boolean inplace, Tensor X, float C) { return linear(inplace, 1.0f / C, X, 0.0f); }
    public Tensor invert_color(boolean inplace, Tensor X) { return linear(inplace, -1.0f, X, 255.0f); }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear(boolean inplace, float alpha, Tensor X, float beta) {
        if(eg.check) { eg.require_int8(X, "X"); }
        Tensor Y = (inplace? X : eg.empty_int8(X.dim).c());
        Syncer sc = eg.core.img_linear2D(Y.address, 
                alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(eg.sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="image: linear2_row">
    public Tensor add_row(boolean inplace, Tensor X1, Tensor X2) { return linear2_row(inplace, X1, X2, 1.0f, 1.0f, 0.0f); }
    public Tensor sub_row(boolean inplace, Tensor X1, Tensor X2) { return linear2_row(inplace, X1, X2, 1.0f, -1.0f, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor linear2_row(boolean inplace, Tensor X1, Tensor X2,
            float alpha, float beta, float gamma) 
    {
        if(eg.check) { 
            eg.require_int8(X1, "X1"); eg.require_dtype(X2, "X2"); 
            eg.check_row(X1, "X1", X2, "X2");
        }
        Tensor Y = (inplace? X1 : eg.empty_int8(X1.dim).c());
        Syncer sc = eg.core.img_linear_dual2D_row(Y.address, 
                X1.address, 
                X2.address, X2.lengthv,
                alpha, beta, gamma,
                X1.lengthv, X1.lastDim());
        if(eg.sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="image: linear2_field">
    public Tensor add_field(boolean inplace, Tensor X1, Tensor X2) {
        return linear2_field(inplace, X1, X2, 1.0f, 1.0f, 0.0f);
    }
    public Tensor sub_field(boolean inplace, Tensor X1, Tensor X2) {
        return linear2_field(inplace, X1, X2, 1.0f, -1.0f, 0.0f);
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor linear2_field(boolean inplace, Tensor X1, Tensor X2,
            float alpha, float beta, float gamma)
    {
        if(eg.check) { 
            eg.require_int8(X1, "X1"); eg.require_dtype(X2, "X2"); 
            eg.check_field(X1, "X1", X2, "X2");
        }
        Tensor Y = (inplace? X1 : eg.empty_int8(X1.dim).c());
        Syncer sc = eg.core.img_linear_dual2D_field(Y.address, 
                X1.address, 
                X2.address, X2.length,//X2.length = field_length
                alpha, beta, gamma, 
                X1.lengthv, X1.lastDim());
        if(eg.sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: quadratic">
    public Tensor square(boolean inplace, Tensor X) { return quadratic(inplace, X, 1.0f, 0.0f, 0.0f); }
    @Passed("CudaFloat32EngieBase")
    public Tensor quadratic(boolean inplace, Tensor X, float alpha, float beta, float gamma) {
        if(eg.check) { eg.require_int8(X, "X"); }
        Tensor Y = (inplace? X : eg.empty_int8(X.dim).c());
        Syncer sc = eg.core.img_quadratic2D(Y.address,
                X.address, alpha, beta, gamma, 
                X.lengthv, X.lastDim());
        if(eg.sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: log, exp">
    public Tensor log(boolean inplace, Tensor X) { return log(inplace, 1.0f, 1.0f, X, 0.0f); }
    public Tensor log(boolean inplace, float alpha, Tensor X, float beta) {  return log(inplace, 1.0f, alpha, X, beta); }
    @Passed("CudaFloat32EngieBase")//Y = C*log(alpha*X + beta)
    public Tensor log(boolean inplace, float C, float alpha, Tensor X, float beta) {
        if(eg.check) { eg.require_int8(X, "X"); }
        Tensor Y = (inplace? X : eg.empty_int8(X.dim).c());
        Syncer sc = eg.core.img_log2D(Y.address,
                C, alpha, X.address, beta,
                X.lengthv, X.lastDim());
        if(eg.sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    public Tensor exp(boolean inplace, Tensor X) { return exp(inplace, 1.0f, X, 0.0f, 0.0f); }
    public Tensor exp(boolean inplace, float alpha, Tensor X, float beta) { return exp(inplace, alpha, X, beta, 0.0f); }
    @Passed("CudaFloat32EngieBase")//Y = exp(alpha*X + beta) + C
    public Tensor exp(boolean inplace, float alpha, Tensor X, float beta, float C) {
        if(eg.check) { eg.require_int8(X, "X"); }
        Tensor Y = (inplace? X : eg.empty_int8(X.dim).c());
        Syncer sc = eg.core.img_exp2D(Y.address, 
                alpha, X.address, beta, C, 
                X.lengthv, X.lastDim());
        if(eg.sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: tensor-trick functions">
    //<editor-fold defaultstate="collapsed" desc="image: pad">
    public Tensor pad2D(boolean inplace, Tensor X, int...p) {
        p = Vector.append(p, 0);//[0..., p, 0]
        return pad(inplace, X, p, Vector.arrayCopy(p));
    }
    public Tensor pad2D(boolean inplace, Tensor X, int[] p0, int[] p1) {
        p0 = Vector.append(p0, 0);//[0..., p0, 0]
        p1 = Vector.append(p1, 0);//[0..., p1, 0]
        return pad(inplace, X, p0, p1);
    }
    
    public Tensor pad(boolean inplace, Tensor X, int...p) { return pad(inplace, X, p, Vector.arrayCopy(p)); }
    @Passed("CudaFloat32EngieBase")
    public Tensor pad(boolean inplace, Tensor X, int[] p0, int[] p1) {
        int[] Xdim = X.dim; int ndim = Xdim.length;
        if(eg.check) {
            eg.require_int8(X, "X");
            eg.must_greater_equal(ndim, "X.ndim", 3);
            if(p0 != null) {
                eg.must_smaller_equal(p0.length, "p0.length", 3);
                eg.must_non_negative(p0, "p0");
            }
            if(p1 != null) {
                eg.must_smaller_equal(p1.length, "p1.length", 3);
                eg.must_non_negative(p1, "p1");
            }
        }
        
        p0 = Vector.expand_from_head(p0, 3);//expand p0 (can be null) to ndim with 0s, [0..., p0]
        p1 = Vector.expand_from_head(p1, 3);//expand p1 (can be null) to ndim with 0s, [0..., p1]
        
        //------[determine Y.dim]-----------------------------------------------
        int[] Ydim = new int[ndim]; int m = ndim - 3;//m >= 0
        for(int i=0; i<m; i++) Ydim[i] = Xdim[i];
        for(int i=m; i<ndim; i++) Ydim[i] = p0[i] + Xdim[i] + p1[i];//[H, W, C]
        Tensor Y = eg.zeros_int8(Ydim);
        //------[determine Y.dim]-----------------------------------------------
         
        int N = X.length / (Xdim[ndim - 1] * Xdim[ndim - 2] * Xdim[ndim - 3]);
         
        Syncer sc1 = eg.core.img_pad(
                Y.c().address, Ydim[ndim - 3], Ydim[ndim - 2], Ydim[ndim - 1],
                X.address    , Xdim[ndim - 3], Xdim[ndim - 2], Xdim[ndim - 1],
                N ,p0[0], p0[1], p0[2]);
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }

        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="image: trim">
    public Tensor trim2D(boolean inplace, Tensor X, int... t) {
        t = Vector.append(t, 0);//[0..., t, 0]
        return trim(inplace, X, t, Vector.arrayCopy(t));
    }
    public Tensor trim2D(boolean inplace, Tensor X, int[] t0, int[] t1) {
        t0 = Vector.append(t0, 0);//[0..., p0, 0]
        t1 = Vector.append(t1, 0);//[0..., p1, 0]
        return trim(inplace, X, t0, t1);
    }
    
    public Tensor trim(boolean inplace, Tensor X, int... t) { return trim(inplace, X, t, Vector.arrayCopy(t)); }
    @Passed("CudaFloat32EngieBase")
    public Tensor trim(boolean inplace, Tensor X, int[] t0, int[] t1) {
        int[] Xdim = X.dim; int ndim = Xdim.length;
        if(eg.check) {
            eg.require_int8(X, "X");
            eg.must_greater_equal(ndim, "X.ndim", 3);
            if(t0 != null) {
                eg.must_smaller_equal(t0.length, "t0.length", 3);
                eg.must_non_negative(t0, "t0");
            }
            if(t1 != null) {
                eg.must_smaller_equal(t1.length, "t1.length", 3);
                eg.must_non_negative(t1, "t1");
            }
        }
        
        t0 = Vector.expand_from_head(t0, 3);//expand t0 (can be null) to ndim with 0s,  [0..., t0]
        t1 = Vector.expand_from_head(t1, 3);//expand t1 (can be null) to ndim with 0s,  [0..., t1]
        
        //------[determine Y.dim]-----------------------------------------------
        int[] Ydim = new int[ndim]; int m = ndim - 3;//m >= 0
        for(int i=0; i<m; i++) Ydim[i] = Xdim[i];
        for(int i=m; i<ndim; i++) Ydim[i] = Xdim[i] - t0[i] - t1[i];//[H, W, C]
        Tensor Y = eg.zeros_int8(Ydim);
        //------[determine Y.dim]-----------------------------------------------
       
        int N = X.length / (Xdim[ndim - 1] * Xdim[ndim - 2] * Xdim[ndim - 3]);
        Syncer sc1 = eg.core.img_trim(
                Y.c().address, Ydim[ndim - 3], Ydim[ndim - 2], Ydim[ndim - 1], 
                X.address    , Xdim[ndim - 3], Xdim[ndim - 2], Xdim[ndim - 1], 
                N, t0[0], t0[1], t0[2]);
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }

        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="auxilary: img_start_from center">
    protected int[] img_start_from_center(int[] small_dim, int[] big_dim) {
        int ndim = small_dim.length;
        int[] arr = new int[] {
            (big_dim[ndim - 3] - small_dim[ndim - 3]) >> 1,
            (big_dim[ndim - 2] - small_dim[ndim - 2]) >> 1,
            (big_dim[ndim - 1] - small_dim[ndim - 1]) >> 1,
        };
        if(arr[0] < 0) arr[0] = 0;
        if(arr[1] < 0) arr[1] = 0;
        if(arr[2] < 0) arr[2] = 0;
        return arr;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="image: expand">
    public Tensor expand2D(boolean inplace, Tensor X, int... out_dim) {
        out_dim = Vector.append(out_dim, -1);//[Xdim, .... out_dim, Xdim(-1)]
        return expand(inplace, X, from_center, out_dim);
    }
    public Tensor expand2D(boolean inplace, Tensor X, int[] start_point, int[] out_dim) {
        start_point = Vector.append(start_point, 0);//[0..., start_point, 0]
        out_dim = Vector.append(out_dim, -1);//[Xdim, .... out_dim, Xdim(-1)]
        return expand(inplace, X, start_point, out_dim);
    }
    
    public Tensor expand(boolean inplace, Tensor X, int... out_dim) { return expand(inplace, X, from_center, out_dim); }
    @Passed("CudaFloat32EngieBase")
    public Tensor expand(boolean inplace, Tensor X, int[] start_point, int[] out_dim) {
        int[] Xdim = X.dim; int ndim = Xdim.length;
        if(eg.check) {
            eg.require_int8(X, "X");
            eg.must_greater_equal(ndim, "X.ndim", 3);
            if(out_dim != null) eg.must_smaller_equal(out_dim.length, "out_dim.length", 3);
            if(start_point != null) {
                eg.must_smaller_equal(start_point.length, "start_point.length", 3);
                eg.must_non_negative(start_point, "start_point");
            }
        }
        
        //------[determine Y.dim]-----------------------------------------------
        int[] Ydim = Vector.expand_from_head_positive(out_dim, X.dim);//[Xdim..., out_dim], Ydim >= 0
        int[] p0 = (start_point == from_center ?         //t0 >= 0
                img_start_from_center(Xdim, Ydim) :      //[(Ydim - Xdim) / 2...]
                Vector.expand_from_head(start_point, 3));//[0..., start_point]
        
        int i0 = ndim - 3, y0 = Xdim[i0] + p0[0]; if(Ydim[i0] < y0) Ydim[i0] = y0;
        int i1 = ndim - 2, y1 = Xdim[i1] + p0[1]; if(Ydim[i1] < y1) Ydim[i1] = y1;
        int i2 = ndim - 1, y2 = Xdim[i2] + p0[2]; if(Ydim[i2] < y2) Ydim[i2] = y2;
        Tensor Y = eg.zeros_int8(Ydim);
        //------[determine Y.dim]-----------------------------------------------
        
        int N = X.length / (Xdim[ndim - 1] * Xdim[ndim - 2] * Xdim[ndim - 3]);
        
        Syncer sc1 = eg.core.img_pad(
                Y.c().address, Ydim[ndim - 3], Ydim[ndim - 2], Ydim[ndim - 1],
                X.address    , Xdim[ndim - 3], Xdim[ndim - 2], Xdim[ndim - 1],
                N, p0[0], p0[1], p0[2]);
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }

        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="image: crop">
    public Tensor crop2D(boolean inplace, Tensor X, int... out_dim) {
        out_dim = Vector.append(out_dim, -1);//[Xdim, .... out_dim, Xdim(-1)]
        return crop(inplace, X, from_center, out_dim);
    }
    public Tensor crop2D(boolean inplace, Tensor X, int[] start_point, int[] out_dim) {
        start_point = Vector.append(start_point, 0);//[0..., start_point, 0]
        out_dim = Vector.append(out_dim, -1);//[Xdim, .... out_dim, Xdim(-1)]
        return crop(inplace, X, start_point, out_dim);
    }
    
    public Tensor crop(boolean inplace, Tensor X, int...out_dim) { return crop(inplace, X, from_center, out_dim);  }
    @Passed("CudaFloat32EngieBase")
    public Tensor crop(boolean inplace, Tensor X, int[] start_point, int[] out_dim) {
        int[] Xdim = X.dim; int ndim = Xdim.length;
        if(eg.check) {
            eg.require_int8(X, "X"); 
            eg.must_greater_equal(ndim, "X.ndim", 3);
            if(out_dim != null) eg.must_smaller_equal(out_dim.length, "out_dim.length", 3);
            if(start_point != null) {
                eg.must_smaller_equal(start_point.length, "start_point.length", 3);
                eg.must_non_negative(start_point, "start_point");
            }
        }
        
        //------[determine Y.dim]-----------------------------------------------
        int[] Ydim = Vector.expand_from_head_positive(out_dim, X.dim);//[Xdim..., out_dim], Ydim >= 0
        int[] t0 = (start_point == from_center ?         //t0 >= 0
                img_start_from_center(Ydim, Xdim) :      //[(Ydim - Xdim) / 2...]
                Vector.expand_from_head(start_point, 3));//[0..., start_point]
        
        int i0 = ndim - 3, y0 = Xdim[i0] - t0[0]; if(Ydim[i0] > y0) Ydim[i0] = y0;
        int i1 = ndim - 2, y1 = Xdim[i1] - t0[1]; if(Ydim[i1] > y1) Ydim[i1] = y1;
        int i2 = ndim - 1, y2 = Xdim[i2] - t0[2]; if(Ydim[i2] > y2) Ydim[i2] = y2;
        
        Tensor Y = eg.zeros_int8(Ydim);
        //------[determine Y.dim]-----------------------------------------------

        int N = X.length / (Xdim[ndim - 1] * Xdim[ndim - 2] * Xdim[ndim - 3]);
        
        Syncer sc1 = eg.core.img_trim(
                Y.c().address, Ydim[ndim - 3], Ydim[ndim - 2], Ydim[ndim - 1], 
                X.address    , Xdim[ndim - 3], Xdim[ndim - 2], Xdim[ndim - 1],
                N, t0[0], t0[1], t0[2]);
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }

        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: transpose(2D -> 4D): X -> X^T">
    @Passed("CudaFloat32EngieBase")
    public Tensor transpose(boolean inplace, Tensor X, int dimIdx1, int dimIdx2) {
        final int[] Xdim = X.dim;
        if(dimIdx1 < 0) dimIdx1 = Xdim.length + dimIdx1;
        if(dimIdx2 < 0) dimIdx2 = Xdim.length + dimIdx2;
        
        if(eg.check) {
            eg.require_int8(X, "X");
            eg.must_greater_equal(Xdim.length, "X.ndim", 2);
            eg.must_smaller_equal(dimIdx1, "dimIdx1", Xdim.length, "X.ndim");
            eg.must_smaller_equal(dimIdx2, "dimIdx2", Xdim.length, "X.ndim");
        }
        
        int[] Ydim = Vector.arrayCopy(Xdim);
        int t = Ydim[dimIdx1]; Ydim[dimIdx1] = Ydim[dimIdx2]; Ydim[dimIdx2] = t;
        Tensor Y = eg.empty_int8(Ydim).c();//Y[newDim, newAddress]
        
        Syncer sc1 = eg.core.img_transpose(
                Y.address, Ydim, 
                X.address, Xdim,
                dimIdx1, dimIdx2,
                X.lastDim(), Y.lastDim(),
                X.length);
      
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: resize">
    public Tensor resize(boolean inplace, Tensor X, float scale) { return resize(inplace, X, scale, scale); }
    public Tensor resize(boolean inplace, Tensor X, float scaleY, float scaleX) {
        int OH = (int) (scaleY * X.dim(-3));//[N, H, W, C]
        int OW = (int) (scaleX * X.dim(-2));
        return resize(inplace, X, OH, OW);
    }
    
    public Tensor resize(boolean inplace, Tensor X, int out_size) { return resize(inplace, X, out_size, out_size); }
    @Passed("CudaFloat32EngieBase")
    public Tensor resize(boolean inplace, Tensor X, int OH, int OW) {
        if(eg.check) {
            eg.require_int8(X, "X");
            eg.must_greater_equal(X.ndim(), "X.ndim", 3);
        }
        
        int[] Xdim = X.dim; int ndim = Xdim.length;
        int N = (ndim == 3 ? 1 : Xdim[0]);
        int IH = Xdim[ndim - 3], IW = Xdim[ndim - 2], C = Xdim[ndim - 1];
       
        int[] Ydim = Vector.arrayCopy(Xdim);//[IH, IW, C] -> [OH, OW, C]
        Ydim[ndim - 3] = OH; Ydim[ndim - 2] = OW;
        Tensor Y = eg.empty_int8(Ydim).c();
       
        Syncer sc1 =  eg.core.img_resize(
                X.address, IH, IW, 
                Y.address, OH, OW, 
                N, C);
       
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }
       
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="image: affine_transformation">
    //<editor-fold defaultstate="collapsed" desc="auxilaury: img_affine">
    protected int[] def_img_affine_outSize(int IH, int IW,
            float m00, float m01, float m02,
            float m10, float m11, float m12) 
    {
        //(0, 0) -> (oh0, ow0), (IH, IW) -> (oh3, ow3)
        float ow0 = m02, ow3 = m00 * IW + m01 * IH + m02;
        float oh0 = m12, oh3 = m10 * IW + m11 * IH + m12;

        //(0, IW) -> (oh1, ow1), (IH, 0) -> (oh2, ow2)
        float ow1 = m00 * IW + m02, ow2 = m01 * IH + m02;
        float oh1 = m10 * IW + m12, oh2 = m11 * IH + m12;

        int OH = (int) Math.ceil(Vector.maxValue(oh0, oh1, oh2, oh3));
        int OW = (int) Math.ceil(Vector.maxValue(ow0, ow1, ow2, ow3));
        return new int[]{ OH, OW };
    }

    protected static float[] invert_affine_matrix(
            float m00, float m01, float m02,
            float m10, float m11, float m12) 
    {
        float det = m00 * m11 - m01 * m10;
        if(Math.abs(det) < Float.MIN_VALUE) throw new IllegalArgumentException(
                "the matrix of affine transformation must be invertable");

        float r00 =  m11 / det, r01 = -m01 / det, r02 = (m01 * m12 - m11 * m02) / det;
        float r10 = -m10 / det, r11 =  m00 / det, r12 = (m10 * m02 - m00 * m12) / det;
        return new float[]{r00, r01, r02, r10, r11, r12}; 
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="img_affine: extended">
    public Tensor translate(boolean inplace, Tensor X, float ty, float tx) { return translate(inplace, X, -1, -1, ty, tx); }
    public Tensor translate(boolean inplace, Tensor X, int OH, int OW, float ty, float tx) {
        return affine(inplace, X, OH, OW, //[1, 0, tx]   [1, 0, 0]
                1.0f, 0.0f, tx,           //[0, 1, ty] * [0, 1, 0]
                0.0f, 1.0f, ty);          //[0, 0,  1]   [0, 0, 1] 
    }
    
    public Tensor scale(boolean inplace, Tensor X, float sy, float sx) { return scale(inplace, X, -1, -1, sy, sx); }
    public Tensor scale(boolean inplace, Tensor X, int OH, int OW, float sy, float sx) {
        return affine(inplace, X, OH, OW,//[sx,  0, 0]   [1, 0, 0]
                sx, 0.0f, 0.0f,          //[ 0, sy, 0] * [0, 1, 0]
                0.0f, sy, 0.0f);         //[ 0,  0, 1]   [0, 0, 1]
    }
    
    public Tensor horizontal_flip(boolean inplace, Tensor X) { return horizontal_flip(inplace, X, -1, -1);  }
    public Tensor horizontal_flip(boolean inplace, Tensor X, int OH, int OW) { 
        int IW = X.dim(-2);//[N, IH, IW, IC]
        return affine(inplace, X, OH, OW,//[-1, 0, IW]   [1, 0, 0]
                -1.0f, 0.0f,  IW,        //[ 0, 1,  0] * [0, 1, 0]
                 0.0f, 1.0f, 0.0f);      //[ 0, 0,  1]   [0, 0, 1]
    }
    
    public Tensor vertical_flip(boolean inplace, Tensor X) { return  vertical_flip(inplace, X, -1, -1); }
    public Tensor vertical_flip(boolean inplace, Tensor X, int OH, int OW) {
        int IH = X.dim(-3);//[N, IH, IW, IC]
        return affine(inplace, X, OH, OW, //[1,  0,  0]   [1, 0, 0]
                 1.0f,  0.0f, 0.0f,       //[0, -1, IH] * [0, 1, 0]
                 0.0f, -1.0f,  IH);       //[0,  0,  1]   [0, 0, 1]
    }
    
    public Tensor shear(boolean inplace, Tensor X, float shy, float shx) { return shear(inplace, X, -1, -1, shy, shx); } 
    public Tensor shear(boolean inplace, Tensor X, int OH, int OW, float shy, float shx) {
        return affine(inplace, X, OH, OW,//[  1, shx, 0]   [1, 0, 0]
                1.0f, shx, 0.0f,         //[shy,   1, 0] * [0, 1, 0]
                shy, 1.0f, 0.0f);        //[  0,   0, 1]   [0, 0, 1]
    } 
    
    public Tensor rotate(boolean inplace, Tensor X, float theta) { return rotate(inplace, X, -1, -1, theta); }
    public Tensor rotate(boolean inplace, Tensor X, int OH, int OW, float theta) {
        float sinx = (float) Math.sin(theta);
        float cosx = (float) Math.cos(theta);
        return affine(inplace, X, OH, OW,//[cosx, -sinx, 0]
                cosx, -sinx, 0.0f,       //[sinx,  cosx, 0]
                sinx,  cosx, 1.0f);      //[   0,     0, 1]
    }
    
    public Tensor rotate(boolean inplace, Tensor X, float theta, float cy, float cx) {
        return rotate(inplace, X, -1, -1, theta, cy, cx);
    }
    public Tensor rotate(boolean inplace, Tensor X, int OH, int OW, 
            float theta, float cy, float cx)  
    {
        float sinx = (float) Math.sin(theta);
        float cosx = (float) Math.cos(theta);
        float a00 = cosx, a01 = -sinx, a02 = cosx*cx - sinx*cy - cx;
        float a10 = sinx, a11 =  cosx, a12 = sinx*cx + cosx*cy - cy;
        return affine(inplace, X, OH, OW,//[1, 0, tx]   [cosx, -sinx, 0]   [1, 0, -tx]   [a00, a01, a02]
                a00, a01, a02,           //[0, 1, ty] * [sinx,  cosx, 0] * [0, 1, -ty] = [a10, a11, a12]
                a10, a11, a12);          //[0, 0,  1]   [   0,     0, 1]   [0, 0,   1]   [  0,  0,    1]
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="img_inverse_affine: M{ m00, m01, m02, m10, m11, m12 }">
    public Tensor inverse_affine(boolean inplace, Tensor X, float[] M) { return inverse_affine(inplace, X, -1, -1, M); }
    public Tensor inverse_affine(boolean inplace, Tensor X,  int OH, int OW, float[] M) {
        if(M == null) throw new NullPointerException("matrix for affine-transformation is null");
        if(M.length != 6) throw new IllegalArgumentException(String.format("M.length { got %d } != 6", M.length));
        return inverse_affine(inplace, X, OH, OW, M[0], M[1], M[2], M[3], M[4], M[5]);
    }
    
    public Tensor inverse_affine(boolean inplace, Tensor X, 
            float m00, float m01, float m02, 
            float m10, float m11, float m12) {
        return inverse_affine(inplace, X, -1, -1, m00, m01, m02, m10, m11, 12);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor inverse_affine(boolean inplace, Tensor X, int OH, int OW, 
            float m00, float m01, float m02, 
            float m10, float m11, float m12)
    {
         if(eg.check) { 
            eg.require_int8(X, "X");
            eg.must_greater_equal(X.ndim(), "X.ndim", 3);
        }
        
        int[] Xdim = X.dim; int ndim = Xdim.length;
        int IH = Xdim[ndim - 3], IW = Xdim[ndim - 2], IC  = Xdim[ndim - 1];
        int N = (ndim == 3 ? 1 :  X.length / (IH * IW * IC));//[N, IH, IW, IC]
        
        if(OH == 0) OH = X.dim(-2);
        if(OW == 0) OW = X.dim(-1);
        if(OH == -1 || OW == -1) {//default out_size of affine transformation
            float[] R = invert_affine_matrix(m00, m01, m02, m10, m11, m12);
            int[] out_size = def_img_affine_outSize(IH, IW, R[0], R[1], R[2], R[3], R[4], R[5]);
            if(OH == -1) OH = out_size[0];
            if(OW == -1) OW = out_size[1];
        }
        
        int[] Ydim = Vector.arrayCopy(Xdim);//[IH, IW, C] -> [OH, OW, C]
        Ydim[ndim - 3] = OH; Ydim[ndim - 2] = OW;
        Tensor Y = eg.empty_int8(Ydim);
        
        Syncer sc1 = eg.core.img_affine(
                X.address, IH, IW,
                Y.c().address, OH, OW,
                m00, m01, m02,
                m10, m11, m12,
                N, IC);
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="img_affine: M{ m00, m01, m02, m10, m11, m12 }"> 
    public Tensor affine(boolean inplace, Tensor X, float[] M) { return affine(inplace, X, -1, -1, M); }
    public Tensor affine(boolean inplace, Tensor X, int OH, int OW, float[] M) {
        if(M == null) throw new NullPointerException("matrix for affine-transformation is null");
        if(M.length != 6) throw new IllegalArgumentException(String.format("M.length { got %d } != 6", M.length));
        return affine(inplace, X, OH, OW, M[0], M[1], M[2], M[3], M[4], M[5]);
    }
   
    public Tensor affine(boolean inplace, Tensor X, 
            float m00, float m01, float m02, 
            float m10, float m11, float m12) {
        return affine(inplace, X, -1, -1, m00, m01, m02, m10, m11, m12);
    }
    @Passed("CudaFloat32EngieBase")
    public Tensor affine(boolean inplace, Tensor X, int OH, int OW, 
            float m00, float m01, float m02, 
            float m10, float m11, float m12)
    {
        if(eg.check) { 
            eg.require_int8(X);
            eg.must_greater_equal(X.ndim(), "X.ndim", 3);
        }
        
        int[] Xdim = X.dim; int ndim = Xdim.length;
        int IH = Xdim[ndim - 3], IW = Xdim[ndim - 2], IC  = Xdim[ndim - 1];
        int N = (ndim == 3 ? 1 :  X.length / (IH * IW * IC));//[N, IH, IW, IC]
        
        if(OH == 0) OH = X.dim(-2);
        if(OW == 0) OW = X.dim(-1);
        if(OH == -1 || OW == -1) {//default out_size of affine transformation
            int[] out_size = def_img_affine_outSize(IH, IW, m00, m01, m02, m10, m11, m12);
            if(OH == -1) OH = out_size[0];
            if(OW == -1) OW = out_size[1];
        }
        
        int[] Ydim = Vector.arrayCopy(Xdim);//[IH, IW, C] -> [OH, OW, C]
        Ydim[ndim - 3] = OH; Ydim[ndim - 2] = OW;
        Tensor Y = eg.empty_int8(Ydim);
        
        float[] R = invert_affine_matrix(m00, m01, m02, m10, m11, m12);
        Syncer sc1 = eg.core.img_affine(
                X.address, IH, IW,
                Y.c().address, OH, OW,
                R[0], R[1], R[2],//r00, r01, r02
                R[3], R[4], R[5],//r10, r11r12
                N, IC);
        
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }
        
        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class: ImageAffiner">
    public static class ImageAffiner 
    {
        private float m00 = 1.0f, m01 = 0.0f, m02 = 0.0f;
        private float m10 = 0.0f, m12 = 1.0f, m11 = 1.0f;
      
        //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
        public final float m00() { return m00; }
        public final ImageAffiner m00(float m00) { this.m00 = m00; return this; }
        
        public final float m01() { return m01; }
        public final ImageAffiner m01(float m01) { this.m01 = m01; return this; }
        
        public final float m02() { return m02; }
        public final ImageAffiner m02(float m02) { this.m02 = m02; return this; }
        
        public final float m10() { return m10; }
        public final ImageAffiner m10(float m10) { this.m10 = m10; return this; }
        
        public final float m11() { return m11; }
        public final ImageAffiner m11(float m11) { this.m11 = m11; return this; }
        
        public final float m12() { return m12; }
        public final ImageAffiner m12(float m12) { this.m12 = m12; return this; }
        
        public final float[] matrix() { return new float[] { m00, m01, m02, m10, m11, m12 }; }
        public final ImageAffiner matrix(float... M) {
            if(M == null || M.length != 6) throw new IllegalArgumentException(
                    "the Matrix for image-affine-transformation must has 6 elements");
            m00 = M[0]; m01 = M[1]; m02 = M[2];
            m10 = M[3]; m11 = M[4]; m12 = M[5];
            return this;
        }
        
        public void append(StringBuilder sb) {
            sb.append(getClass().getSimpleName()).append("{");
            sb.append("\n[").append(m00).append(", ").append(m01).append(", ").append(m02).append("]");
            sb.append("\n[").append(m10).append(", ").append(m11).append(", ").append(m12).append("]");
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder(128);
            this.append(sb);
            return sb.toString();
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="affine_operations">
        public ImageAffiner copy() {
            ImageAffiner af = new ImageAffiner();
            af.m00 = m00; af.m01 = m01; af.m02 = m02;
            af.m10 = m10; af.m11 = m11; af.m12 = m12;
            return af;
        }
        
        public ImageAffiner to_indentity() {
            m00 = 1.0f; m01 = 0.0f; m02 = 0.0f;
            m10 = 0.0f; m12 = 1.0f; m11 = 1.0f;
            return this;
        }
        
        public ImageAffiner translate(float ty, float tx) {
            m02 = m02 + tx;//[1, 0, tx]   [m00, m01, m02]  
            m12 = m12 + ty;//[0, 1, ty] * [m10, m11, m12]
            return this;   //[0, 0,  1]   [  0,   0,   1]
        } 
        
        public ImageAffiner scale(float sy, float sx) {
            m00 = m00 * sx; m01 = m01 * sx; m02 = m02 * sx;//[sx,  0, 0]   [m00, m01, m02]
            m10 = m10 * sy; m11 = m11 * sy; m12 = m12 * sy;//[ 0, sy, 0] * [m10, m11, m12]
            return this;                                   //[ 0,  0, 1]   [  0,   0,   1]
        }
        
        public ImageAffiner horizontal_flip(int width) {
            m00 = -m00; m01 = -m01;//[-1, 0, IW]   [m00, m01, m02] 
            m02 = -m02 + width;    //[ 0, 1,  0] * [m10, m11, m12]
            return this;           //[ 0, 0,  1]   [  0,   0,   1]
        }
        
        public ImageAffiner vertical_flip(int height) {
            m10 = -m10; m11 = -m11;//[1,  0,  0]   [m00, m01, m02] 
            m12 = -m12 + height;   //[0, -1, IH] * [m10, m11, m12]
            return this;           //[0,  0,  1]   [  0,   0,   1]
        }
        
        public ImageAffiner shear(float shy, float shx) {
            float M00 = m00 + shx * m10, M10 = shy * m00 + m10;//[  1, shx, 0]   [m00, m01, m02]
            float M01 = m01 + shx * m11, M11 = shy * m01 + m11;//[shy,   1, 0] * [m10, m11, m12]
            float M02 = m02 + shx * m12, M12 = shy * m02 + m12;//[  0,   0, 1]   [  0,   0,   1]
            m00 = M00; m01 = M01; m02 = M02;
            m10 = M10; m11 = M11; m12 = M12;
            return this;
        }
        
        public ImageAffiner rotate(float theta) {
            double sinx = Math.sin(theta), cosx = Math.cos(theta);
            double M00 = cosx * m00 - sinx * m10, M10 = sinx * m00 + cosx * m10;//[cosx, -sinx, 0]   [m00, m01, m02]
            double M01 = cosx * m01 - sinx * m11, M11 = sinx * m01 + cosx * m11;//[sinx,  cosx, 0] * [m10, m11, m12]
            double M02 = cosx * m02 - sinx * m12, M12 = sinx * m02 + cosx * m12;//[   0,     0, 1]   [  0,   0,   1]
            m00 = (float) M00; m01 = (float) M01; m02 = (float) M02;
            m10 = (float) M10; m11 = (float) M11; m12 = (float) M12;
            return this;
        }
        
        public ImageAffiner rotate(float theta, float cy, float cx) {
            double sinx = Math.sin(theta), cosx = Math.cos(theta);
            //[1, 0, tx]   [cosx, -sinx, 0]   [1, 0, -tx]   [a00, a01, a02]
            //[0, 1, ty] * [sinx,  cosx, 0] * [0, 1, -ty] = [a10, a11, a12]
            //[0, 0,  1]   [   0,     0, 1]   [0, 0,   1]   [  0,  0,    1]
            double a00 = cosx, a01 = -sinx, a02 = cosx*cx - sinx*cy - cx;
            double a10 = sinx, a11 =  cosx, a12 = sinx*cx + cosx*cy - cy;
            //[a00, a01, a02]   [m00, m01, m02]
            //[a10, a11, a12] * [m10, m11, m12]
            //[  0,  0,    1]   [  0,   0,   1]
            double M00 = a00*m00 + a01*m10, M01 = a00*m01 + a01*m11, M02 = a00*m02 + a01*m12 + a02;
            double M10 = a10*m00 + a11*m10, M11 = a10*m01 + a11*m11, M12 = a10*m02 + a11*m12 + a12;
            m00 = (float) M00; m01 = (float) M01; m02 = (float) M02;
            m10 = (float) M10; m11 = (float) M11; m12 = (float) M12;
            return this;
        }
        
        public ImageAffiner invert() {//find the reverse of M
            float[] R = invert_affine_matrix(m00, m01, m02, m10, m11, m12);
            m00 = R[0]; m01 = R[1]; m02 = R[2];
            m10 = R[3]; m11 = R[4]; m12 = R[5];
            return this;
        }
        //</editor-fold>
        
        public Tensor transform(Tensor X, boolean inplace) {
            return X.eg.img.affine(inplace, X, -1, -1, m00, m01, m02, m10, m11, m12);
        }
        public Tensor transform(Tensor X, boolean inplace, int OH, int OW) {
            return X.eg.img.affine(inplace, X, OH, OW, m00, m01, m02, m10, m11, m12);
        }
    }
    //</editor-fold>
    public ImageAffiner affine() { return new ImageAffiner(); } 
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="img: concat: X[] -> Y">    
    public Tensor concat(Tensor... X) { return concat(-1, X); }
    @Passed("CudaFloat32EngieBase")
    public Tensor concat(int dimIdx, Tensor... X)  {
        X = eg.exclude_null_tensors(X);
        final int ndim = X[0].ndim(); if(dimIdx < 0) dimIdx = ndim + dimIdx;
        
        int[][] dimX = new int[X.length][];
        for(int i=0; i<X.length; i++) dimX[i] = X[i].dim;
        
        if(eg.check) {
            eg.require_int8(X);
            if(X.length < 2) throw new IllegalArgumentException("At least input two tensors to concat");
            for(int i=0; i<dimX.length; i++) 
                if(dimX[i].length <= dimIdx) throw new IllegalArgumentException(
                    "dimIndex exceeds X[" + i + "].ndim");
            for(int i=0; i<dimX.length; i++) 
                if(dimX[i].length != ndim) throw new IllegalArgumentException(String.format(
                        "X[%d].ndim(%d) ! = X[0].ndim(%d)", i, dimX[i].length, ndim));
            for(int i=1; i<X.length; i++) 
                for(int j=0; j<ndim; j++) {
                    if(j == dimIdx) continue;//only dimIndex is different
                    if(dimX[0][j] != dimX[i][j]) throw new IllegalArgumentException(
                            String.format("X[%d].dim[%d](%d) != X[0].dim[%d](%d)",i, j, dimX[i][j], j, dimX[0][j]));
                }
        }
        
        //compute the dim of output tensor--------------------------------------
        int[] dimY = Vector.arrayCopy(dimX[0]);//concat_dim = sum(X[i].dim(dimIndex), 0, n-1) 
        for(int i=1; i<X.length; i++) dimY[dimIdx] += dimX[i][dimIdx];//dimIndex: the concat dim
        Tensor Y = eg.empty_int8(dimY);
        
        //compute the copy params-----------------------------------------------
        int commonWidth = 1;//dimSize multiple: from (dimIndex + 1) to End
        for(int i = dimIdx + 1; i<ndim; i++) commonWidth *= dimX[0][i];
        
        int[] copyWidth = new int[X.length];
        int[] strideX = new int[X.length];
        for(int i=0; i<X.length; i++) {
            copyWidth[i] = commonWidth * dimX[i][dimIdx];//from dimIndex to End
            
            int width = X[i].lastDim();//consider memAlignment: 
            int stride = ((width + 3) >> 2) << 2;
            strideX[i] = copyWidth[i] / width * stride;
            if(dimIdx != ndim - 1) copyWidth[i] = strideX[i];
        } 
        
        int strideY = commonWidth * dimY[dimIdx]; {//from dimIndex to End
            int width = Y.lastDim();//consider mem alignment
            int stride = ((width + 3) >> 2) << 2;
            strideY = strideY / width * stride;
        }
                
        Syncer[] sc = new Syncer[X.length]; Y.c();//Y is synchronized 
        for(int i=0, Ystart = 0; i<X.length; Ystart += copyWidth[i++]) {
            int length = (dimIdx == ndim - 1 ? X[i].length : X[i].lengthv);
            sc[i] = eg.core.img_gappedMemcpy2D(
                    X[i].address, 0, strideX[i], 
                    Y.address, Ystart, strideY, 
                    copyWidth[i], length);
        }
        
        if(eg.sync) { for(Syncer syncer : sc) syncer.sync(); }
        else Y.setSyncer(new ChainSyncer(sc));
        return Y;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="img: split & chunk: X -> Y[]">
    @Passed("CudaFloat32EngieBase")
    public Tensor[] chunk(Tensor X, int dimIdx, int n) {
        int ndim = X.ndim(); if(dimIdx < 0) dimIdx = ndim + dimIdx;
        
        int dimX[] = X.dim, dimSize = dimX[dimIdx]; //dimSize = dimX[dimIndex] = sum(section)
        if(eg.check) {
            eg.require_int8(X, "X");
            if(n < 1) throw new IllegalArgumentException("n at least 2 to chunk a tensor");
            if(n > dimSize) throw new IllegalArgumentException(String.format(
                    "n { got %d }> X.dim[%d] { got %d }", n, dimIdx, dimSize));
        }
        
        int[] section = new int[n];
        int div = dimSize  / n, rm = dimSize % n;
        for(int i=0; i<n; i++) section[i] = div;
        section[n - 1] += rm;
        
        return __split(X, dimIdx, section);
    }
    
    @Passed("CudaFloat32EngieBase") 
    public Tensor[] split(Tensor X, int dimIdx, int...section) {
        int ndim = X.ndim(); if(dimIdx < 0) dimIdx = ndim + dimIdx;
        int dimX[] = X.dim, dimSize = dimX[dimIdx]; //dimSize = dimX[dimIndex] = sum(section)
        int sectionSum = eg.negativeSection(dimSize, section);//exclude -1 in section
         
        if(eg.check) {
            eg.require_int8(X);
            eg.must_greater_equal(section.length, "section.length", 2);
            eg.must_positive(section, "section");
            if(sectionSum != dimSize) throw new IllegalArgumentException(String.format(
                    "sum(section) { got %d } != X.dim[%d] { got %d }", 
                    sectionSum, dimIdx, dimSize));
        }
        
        return __split(X, dimIdx, section);
    }
    
    //<editor-fold defaultstate="collapsed" desc="inner-code: __split">
    Tensor[] __split(Tensor X, int dimIdx, int[] section) {
        //create sub Tensor[] Y based on section--------------------------------
        int dimX[] = X.dim, ndim = X.ndim();
        
        Tensor[] Y = new Tensor[section.length];
        for(int i=0, dimY[] = Vector.arrayCopy(dimX); i<section.length; i++) {
            dimY[dimIdx] = section[i]; 
            Y[i] = eg.empty_int8(dimY);
        }
       
        //compute the copy params-----------------------------------------------
        int commonWidth = 1;//dimSize multiple: from (dimIndex + 1) to End
        for(int i = dimIdx + 1; i<ndim; i++) commonWidth *= dimX[i];
        
        int[] copyWidth = new int[Y.length];
        int[] strideY = new int[Y.length];
        for(int i = 0; i<copyWidth.length; i++){
            copyWidth[i] = commonWidth * section[i];//from dimIndex to End
            
            int width = Y[i].lastDim();//consider memory alginment
            int stride = ((width + 3) >> 2) << 2;
            strideY[i] = copyWidth[i] / width * stride;
            
            //width the same mem_struture, is dimIdex != -1
            if(dimIdx != ndim - 1) copyWidth[i] = strideY[i];
        }
        
        //compute the start index in X(src) for each element of Y[](dst)--------
        int strideX = commonWidth * dimX[dimIdx]; {//from dimIndex to End
            int width = X.lastDim();//consider memAlignment
            int stride = ((width + 3) >> 2) << 2;
            strideX = strideX / width * stride;
        }
        
        Syncer[] scs = new Syncer[Y.length];
        for(int i=0, Xstart = 0; i<Y.length; Xstart += copyWidth[i++]) {
            int length = ((dimIdx == ndim - 1) ? Y[i].length : Y[i].lengthv);
            scs[i] = eg.core.img_gappedMemcpy2D(
                    X.address, Xstart, strideX, 
                    Y[i].c().address, 0, strideY[i],//Y[i] is synchronized
                    copyWidth[i], length);
        }
        
        if(eg.sync) for (Syncer sc : scs) sc.sync();
        else for(int i=0; i<Y.length; i++) Y[i].setSyncer(scs[i]);
        return Y;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: extract_3channels">
    protected int channel_red = 13;
    public int channel_red() { return channel_red; }  
    public ImageEngine channel_red(int cred) {
        eg.must_greater_equal(cred, "red_channel", 0);
        this.channel_red = cred; 
        return this; 
    }
    
    protected int channel_green = 80;
    public int channel_green() { return channel_green; }
    public ImageEngine channel_green(int cgreen) { 
        eg.must_greater_equal(cgreen, "green_channel", 0);
        this.channel_green = cgreen;
        return this; 
    }
    
    protected int channel_blue = 132;
    public int channel_blue() { return channel_blue; }
    public ImageEngine channel_blue(int cblue) { 
        eg.must_greater_equal(cblue, "blue_channel", 0);
        this.channel_blue = cblue; 
        return this; 
    }
    
    public Tensor extract_BGR(boolean inplace, Tensor X) { return extract_3channels(inplace, X, channel_blue, channel_green, channel_red);  }
    public Tensor extract_RGB(boolean inplace, Tensor X) { return extract_3channels(inplace, X, channel_red, channel_green, channel_blue);  }
    @Passed("CudaFloat32EngieBase")
    public Tensor extract_3channels(boolean inplace, Tensor X, int c0, int c1, int c2) {
        int Xdim[] = X.dim, ndim = Xdim.length;
        if(eg.check) { 
            eg.require_int8(X, "X"); 
            eg.must_greater_equal(ndim, "X.ndim", 2);
        }
        
        int[] Ydim = Vector.arrayCopy(X.dim); Ydim[ndim - 1] = 3;
        Tensor Y = eg.empty_int8(Ydim).c();
        
        Syncer sc1 = eg.core.extract_3channels(
                X.address, X.dim(-1), 
                Y.address, c0, c1, c2, 
                Y.length / 3);//lengthv = N*IH*IW
     
        //inplace = false, return the new Tensor Y==============================
        if(!inplace) { if(eg.sync) sc1.sync(); else Y.setSyncer(sc1); return Y; }

        //inplace = true, return the old Tensor X===============================
        long old_memLen = X.mem_size, oldAddr = X.address;
        X.copy_memoryMetaData_and_deleteSrc(Y);//let: X.dim = Y.dim/newDim, X.address = Y.address.newAddress
        
        Syncer sc = Syncer.dual(sc1, ()->{ eg.core.free(old_memLen, oldAddr); });
        if(eg.sync) sc.sync(); else X.setSyncer(sc);
        return X;
    }
    //</editor-fold>
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="image: Tensor -> BufferedImage">
    public BufferedImage BGR(Tensor X) { return BGR(X, channel_blue, channel_green, channel_red); }
    public BufferedImage BGR(Tensor X, int blue, int green, int red) {
        if(eg.check) { eg.must_equal(X.ndim(), "X.ndim", 3); }
        if(X.dim(-1) == 3) { BufferedImage img = cv.BGR(X); return img; }
        Tensor rgb = extract_3channels(false, X, blue, green, red).c();
        BufferedImage img = cv.BGR(rgb); rgb.delete();
        return img;
    }
    
    public BufferedImage RGB(Tensor X) { return RGB(X, channel_red, channel_green, channel_blue); }
    public BufferedImage RGB(Tensor X, int red, int green, int blue) {
        if(eg.check) { eg.must_equal(X.ndim(), "X.ndim", 3); }
        if(X.dim(-1) == 3) { BufferedImage img = cv.RGB(X); return img; }
        Tensor rgb = extract_3channels(false, X, red, green, blue).c();
        BufferedImage img = cv.RGB(rgb); rgb.delete();
        return img;
    }
    
    public BufferedImage gray(Tensor X) {
        if(eg.check) { 
            eg.must_smaller_equal(X.ndim(), "X.ndim", 3);//<=3
            eg.must_greater_equal(X.ndim(), "X.ndim", 2);//>=2
        }
        
        int IH = X.dim(0), IW = X.dim(1);
        if(X.ndim() == 3) {
            Tensor fX = pixel_to_dtype(false, X).c();
            Tensor gray = eg.row_mean(fX, fX.lastDim()).c(); fX.delete();
            gray = dtype_to_pixel(true, gray).c();
            BufferedImage img = cv.gray(gray.pixel(), IH, IW); gray.delete();
            return img;
        }
        return cv.gray(X.pixel(), IH, IW);
    }
    
    public BufferedImage gray(Tensor X, Color black, Color white) {
        if(eg.check) { 
            eg.must_smaller_equal(X.ndim(), "X.ndim", 3);//<=3
            eg.must_greater_equal(X.ndim(), "X.ndim", 2);//>=2
        }
        
        int IH = X.dim(0), IW = X.dim(1);
        if(X.ndim() == 3) {
            Tensor fX = pixel_to_dtype(false, X).c();
            Tensor gray = eg.row_mean(fX, fX.lastDim()).c(); fX.delete();
            gray = dtype_to_pixel(true, gray).c();
            BufferedImage img = cv.gray(gray.pixel(), IH, IW, black, white); gray.delete();
            return img;
        }
        return cv.gray(X.pixel(), IH, IW, black, white);
    }
    
    public BufferedImage[] channel_graphics(Tensor X, Color black, Color white) {
        return cv.channel_graphics(X, black, white);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: adjust_color">
    @Passed("CudaFloat32EngieBase")//<1> factor = -1: black, <2> factor = 0: original, <3> factor = 1: white 
    public Tensor adjust_brightness(boolean inplace, Tensor X, float factor) {
        if(eg.check) { eg.require_int8(X, "X"); }
        if(factor == 0) return (inplace ? X : eg.copy_int8(X));//no need to adjust
        Tensor Y = (inplace? X : eg.empty_int8(X.dim).c());
        Syncer sc = eg.core.img_linear2D(Y.address, 
                1.0f, X.address, factor*255,
                X.lengthv, X.lastDim());
        if(eg.sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")//<1> factor = -1: black and white, <2> factor = 0: original, <3> factor > 0: enhance
    public Tensor adjust_saturation(boolean inplace, Tensor X, float factor) {
        if(eg.check) { eg.require_int8(X, "X"); }
        if(factor == 0) return (inplace ? X : eg.copy_int8(X));//no need to adjust
        
        Tensor fX = pixel_to_dtype(false, X);
        Tensor Y = (inplace? X : eg.empty_int8(X.dim));
        
        Tensor gray = eg.row_mean(fX.c(), X.lastDim()).c();//averge pixel in each channel
        Syncer sc = eg.core.img_linear_dual2D_field(Y.c().address, 
                X.address, 
                gray.address, gray.length,//X2.length = field_length
                (1.0f + factor), -factor, 0.0f, 
                X.lengthv, X.lastDim()); 
        eg.delete(fX);//release fX
        
        if(eg.sync) { sc.sync(); eg.delete(gray); } 
        else Y.setSyncer(Syncer.dual(sc, ()->{ eg.delete(gray); }));
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")//<1> factor = -1: mean of pixel, <2> factor = 0: original, <3> factor > 0: enhance
    public Tensor adjust_constrast(boolean inplace, Tensor X, float factor) {
        if(eg.check) { eg.require_int8(X, "X"); }
        if(factor == 0) return (inplace ? X : eg.copy_int8(X));//no need to adjust
        
        Tensor fX = pixel_to_dtype(false, X);
        Tensor Y = (inplace? X : eg.empty_int8(X.dim));
        
        float mean = fX.c().mean().get();
        float alpha = (factor + 1);
        float beta = (float) (mean * (1 - Math.sqrt(factor + 1)));
        Syncer sc = eg.core.img_linear2D(Y.c().address, 
                alpha, X.address, beta,
                X.lengthv(), X.lastDim());
        eg.delete(fX);
        if(eg.sync) sc.sync(); else Y.setSyncer(sc);
        return Y;
    }
    
    @Passed("CudaFloat32EngieBase")
    public Tensor adjust_color(boolean inplace, Tensor X, float brightness, float saturation, float contrast) {
        if(eg.check) { eg.require_int8(X, "X"); }
        if (brightness == 0f && saturation == 0f && contrast == 0f) return (inplace ? X : eg.copy_int8(X));//no need to adjust
        if (saturation == 0f && contrast   == 0f) return adjust_brightness(inplace, X, brightness);
        if (brightness == 0f && contrast   == 0f) return adjust_saturation(inplace, X, saturation);
        if (brightness == 0f && saturation == 0f) return adjust_constrast (inplace, X, contrast);
        
        Tensor Y = (inplace? X : eg.empty_int8(X.dim));
        Tensor fX = pixel_to_dtype(false, X).c();
        Tensor gray = eg.row_mean(fX.c(), X.lastDim());//averge pixel in each channel
        
        //<1> f1 = brightness: y = x + 255*f1
        //<2> f2 = saturation: y = (1 + f2)*x - f2*gray
        //<3> f3 = contrast:   y = (1 + f3)*x + mean*(1 - sqrt(1 + f3))  
        final float f1 = brightness, f2 = saturation, f3 = contrast;
        float alpha = (1 + f2) * (1 + f3);
        float beta = -f2 * (1 + f3);
        float mean = (f3 == 0 ? 0 : eg.straight_mean(fX).get());
        float gamma = (float) (255*f1*alpha + mean *(1 - (Math.sqrt(1 + f3))));
        
        Syncer sc = eg.core.img_linear_dual2D_field(Y.c().address,
                X.address,//Y = alpha*X + beta*gray + gamma
                gray.c().address, gray.length,
                alpha, beta, gamma,
                X.lengthv, X.lastDim());
        eg.delete(fX);//release fX
        
        if(eg.sync) { sc.sync(); eg.delete(gray); } 
        else Y.setSyncer(Syncer.dual(sc, ()->{ eg.delete(gray); }));
        return Y;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: random_function">
    public ImageEngine set_seed(long seed) { eg.set_seed(seed); return this; }
    protected float next_float() { return eg.core.random().nextFloat(); }
    protected float next_float(float min, float max) { return (max - min) * eg.core.random().nextFloat() + min; }
    
    //<editor-fold defaultstate="collapsed" desc="image: random color">
    public Tensor jit_brightness(float amp, boolean inplace, Tensor X, float factor) {
        factor += next_float(-amp, amp);
        return adjust_brightness(inplace, X, factor);
    }
    
    public Tensor jit_saturation(float amp, boolean inplace, Tensor X, float factor) {
        factor += next_float(-amp, amp);
        return adjust_saturation(inplace, X, factor);
    }
    
    public Tensor jit_constrast(float amp, boolean inplace, Tensor X, float factor) {
        factor += next_float(-amp, amp);
        return adjust_constrast(inplace, X, factor);
    }
    
    public Tensor jit_color(float amp, boolean inplace, Tensor X, float brightness, float saturation, float contrast) {
        brightness += next_float(-amp, amp);
        saturation += next_float(-amp, amp);
        contrast   += next_float(-amp, amp);
        return adjust_color(inplace, X, brightness, saturation, contrast);
    }
    
    public Tensor jit_color(float[] amp, boolean inplace, Tensor X, float brightness, float saturation, float contrast) {
        if(amp == null) throw new NullPointerException();
        if(amp.length != 3) throw new IllegalArgumentException(String.format(
                "amp.length { got %d } must == 3", amp.length));
        
        brightness += next_float(-amp[0], amp[0]);
        saturation += next_float(-amp[1], amp[1]);
        contrast   += next_float(-amp[2], amp[2]);
        return adjust_color(inplace, X, brightness, saturation, contrast);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image: random expand, crop">
    public Tensor random_expand(boolean inplace, Tensor X, int... out_dim) {
        int[] in_dim = { X.dim(-3), X.dim(-2), X.dim(-1) };
        out_dim = Vector.expand_from_head_positive(out_dim, in_dim);
        int[] start_point = new int[]{
            (int)((out_dim[0] - in_dim[0]) * next_float()),//height
            (int)((out_dim[1] - in_dim[1]) * next_float()),//width
            (int)((out_dim[2] - in_dim[2]) * next_float()) //channel
        };
        return expand(inplace, X, start_point, out_dim);
    }
    
    public Tensor random_crop(boolean inplace, Tensor X, int... out_dim) {
        int[] in_dim = { X.dim(-3), X.dim(-2), X.dim(-1) };
        out_dim = Vector.expand_from_head_positive(out_dim, in_dim);
        int[] start_point = new int[]{
            (int)((in_dim[0] - out_dim[0]) * next_float()),//height
            (int)((in_dim[1] - out_dim[1]) * next_float()),//width
            (int)((in_dim[2] - out_dim[2]) * next_float()) //channel
        };
        return crop(inplace, X, start_point, out_dim);
    }
    
    public Tensor random_expand2D(boolean inplace, Tensor X, int... out_dim) {
        int[] in_dim = { X.dim(-3), X.dim(-2) };//[height, width]
        out_dim = Vector.expand_from_head_positive(out_dim, in_dim);
        out_dim = Vector.append(out_dim, X.dim(-1));
        int[] start_point = new int[]{
            (int)((out_dim[0] - in_dim[0]) * next_float()),//height
            (int)((out_dim[1] - in_dim[1]) * next_float()),//width
            0 //don't expand channel
        };
        return expand(inplace, X, start_point, out_dim);
    }
    
    public Tensor random_crop2D(boolean inplace, Tensor X, int... out_dim) {
        int[] in_dim = { X.dim(-3), X.dim(-2) };//[height, width]
        out_dim = Vector.expand_from_head_positive(out_dim, in_dim);
        out_dim = Vector.append(out_dim, X.dim(-1));
        int[] start_point = new int[]{
            (int)((in_dim[0] - out_dim[0]) * next_float()),//height
            (int)((in_dim[1] - out_dim[1]) * next_float()),//width
            0 //don't crop channel
        };
        return crop(inplace, X, start_point, out_dim);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static class RandomImageAffiner">
    public static class RandomImageAffiner extends ImageAffiner
    {
        protected final ImageEngine ieg;
        
        public RandomImageAffiner(ImageEngine ieg) { this.ieg = ieg; }
        
        //<editor-fold defaultstate="collapsed" desc="affine_operations">
        public ImageAffiner random_translate(float amp, float ty, float tx) {
            ty *= 1.0f + ieg.next_float(-amp, amp);
            tx *= 1.0f + ieg.next_float(-amp, amp);
            return translate(ty, tx);
        } 
        
        public ImageAffiner random_scale(float amp, float sy, float sx) {
            sy *= 1.0f + ieg.next_float(-amp, amp);
            sx *= 1.0f + ieg.next_float(-amp, amp);
            return scale(sy, sx);
        }
        
        public ImageAffiner random_horizontal_flip(float p, int width) {
            return (ieg.next_float() <= p? horizontal_flip(width) : this);
        }
        
        public ImageAffiner random_vertical_filp(float p, int height) {
            return (ieg.next_float() <= p? vertical_flip(height) : this);
        }
        
        public ImageAffiner random_shear(float amp, float shy, float shx) {
            shy *= 1.0f + ieg.next_float(-amp, amp);
            shx *= 1.0f + ieg.next_float(-amp, amp);
            return shear(shy, shx);
        }
        
        public ImageAffiner random_rotate(float amp, float theta) {
            theta *= 1.0f + ieg.next_float(-amp, amp);
            return rotate(theta);
        }
        
        public ImageAffiner random_rotate(float amp, float theta, float cy, float cx) {
            theta *= 1.0f + ieg.next_float(-amp, amp);
            cy    *= 1.0f + ieg.next_float(-amp, amp);
            cx    *= 1.0f + ieg.next_float(-amp, amp);
            return rotate(theta, cy, cx);
        }
        //</editor-fold>
    }
    //</editor-fold>
    public RandomImageAffiner random_affine() { return new RandomImageAffiner(this); }
    //</editor-fold>
}
