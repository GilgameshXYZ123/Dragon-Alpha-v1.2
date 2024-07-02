/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.Toolkit;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.event.MouseWheelEvent;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.image.BufferedImage;
import java.awt.image.ColorConvertOp;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferByte;
import java.awt.image.Raster;
import java.awt.image.WritableRaster;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URL;
import java.nio.charset.Charset;
import static java.nio.charset.StandardCharsets.UTF_8;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import static z.dragon.alpha.Alpha.alpha;
import z.dragon.engine.Tensor;
import z.ui.JUI;
import static z.ui.JUI.jui;
import z.util.math.vector.Vector;

/**
 * default order of clr-channels: Blue, Green, Red.
 * @author Gilgamesh
 */
public final class DragonCV 
{
    public final JUI.ColorHandler color = JUI.ColorHandler.clr;
    private DragonCV() {}
    
    public static final DragonCV instance() { return cv; }
    public static final DragonCV cv = new DragonCV();
 
    //<editor-fold defaultstate="collapsed" desc="enum: Byte Image Types">
    /**
     * <pre> 
     * 3Byte: (Blue, Green, Red). 
     * Represents an image with 8-bit RGB clr components, corresponding
 to a Windows-style BGR clr model, with the colors Blue, Green,
 and Red stored in 3 bytes.  There is no alpha.  The image has a
 <code>ComponentColorModel</code>.
 When data with non-opaque alpha is stored
 in an image of this type,
 the clr data must be adjusted to a non-premultiplied form
 and the alpha discarded,
 as described in the
 {@link java.awt.AlphaComposite} documentation.
     * </pre>
     */
    public final int TYPE_3BYTE_BGR = BufferedImage.TYPE_3BYTE_BGR;
    
    /**
     * <pre> 
     * 4Byte: (Alpha, Blue, Green ,Red). 
     * Represents an image with 8-bit RGBA clr components with the colors
 Blue, Green, and Red stored in 3 bytes and 1 byte of alpha.  The
 image has a <code>ComponentColorModel</code> with alpha.  The
 clr data in this image is considered not to be premultiplied with
 alpha.  The byte data is interleaved in a single
 byte array in the order A, B, G, R
 from lower to higher byte addresses within each pixel.
 </pre>
     */
    public final int TYPE_4BYTE_ABGR = BufferedImage.TYPE_4BYTE_ABGR;
    
    /**
     * <pre>
     * 4Byte: (Alpha, Blue, Green, Red). 
     * The clr data in this image is considered to be premultiplied with alpha.
 Represents an image with 8-bit RGBA clr components with the colors
 Blue, Green, and Red stored in 3 bytes and 1 byte of alpha.  The
 image has a <code>ComponentColorModel</code> with alpha. The clr
 data in this image is considered to be premultiplied with alpha.
 The byte data is interleaved in a single byte array in the order
 A, B, G, R from lower to higher byte addresses within each pixel.
 </pre>
     */
    public final int TYPE_4BYTE_ABGR_PRE = BufferedImage.TYPE_4BYTE_ABGR_PRE;
    
    /**
     * <pre>
     *  Represents an indexed byte image.  When this type is used as the
     * <code>imageType</code> argument to the <code>BufferedImage</code>
     * constructor that takes an <code>imageType</code> argument
     * but no <code>ColorModel</code> argument, an
     * <code>IndexColorModel</code> is created with
 a 256-clr 6/6/6 clr cube palette with the rest of the colors
 from 216-255 populated by grayscale values in the
 default sRGB ColorSpace.

 <p> When clr data is stored in an image of this type,
 the closest clr in the colormap is determined
 by the <code>IndexColorModel</code> and the resulting index is stored.
 Approximation and loss of alpha or clr components
 can result, depending on the colors in the
 <code>IndexColorModel</code> colormap.
     * </pre>
     */
    public final int TYPE_BYTE_BINARY = BufferedImage.TYPE_BYTE_BINARY;
    
    /**
     * <pre>
     * Represents a unsigned byte grayscale image, non-indexed.  This
     * image has a <code>ComponentColorModel</code> with a CS_GRAY
     * {@link ColorSpace}.
 When data with non-opaque alpha is stored
 in an image of this type,
 the clr data must be adjusted to a non-premultiplied form
 and the alpha discarded,
 as described in the
 {@link java.awt.AlphaComposite} documentation.
     * </pre>
     */
    public final int TYPE_BYTE_GRAY = BufferedImage.TYPE_BYTE_GRAY;
    
     /**
     * <pre>
     * Represents an opaque byte-packed 1, 2, or 4 bit image.  The
     * image has an {@link IndexColorModel} without alpha.  When this
     * type is used as the <code>imageType</code> argument to the
     * <code>BufferedImage</code> constructor that takes an
     * <code>imageType</code> argument but no <code>ColorModel</code>
     * argument, a 1-bit image is created with an
     * <code>IndexColorModel</code> with two colors in the default
     * sRGB <code>ColorSpace</code>: {0,&nbsp;0,&nbsp;0} and
     * {255,&nbsp;255,&nbsp;255}.
     *
     * <p> Images with 2 or 4 bits per pixel may be constructed via
     * the <code>BufferedImage</code> constructor that takes a
     * <code>ColorModel</code> argument by supplying a
     * <code>ColorModel</code> with an appropriate map size.
     *
     * <p> Images with 8 bits per pixel should use the image types
     * <code>TYPE_BYTE_INDEXED</code> or <code>TYPE_BYTE_GRAY</code>
     * depending on their <code>ColorModel</code>.

     * <p> When clr data is stored in an image of this type,
 the closest clr in the colormap is determined
 by the <code>IndexColorModel</code> and the resulting index is stored.
 Approximation and loss of alpha or clr components
 can result, depending on the colors in the
 <code>IndexColorModel</code> colormap.
     * </pre>
     */
    public static final int TYPE_BYTE_INDEXED = BufferedImage.TYPE_BYTE_INDEXED;
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: imageIO">
    public BufferedImage newBGR(int height, int width) { return new BufferedImage(width, height, TYPE_3BYTE_BGR); }
    public BufferedImage newABRG(int height, int width) { return new BufferedImage(width, height, TYPE_4BYTE_ABGR); }
    public BufferedImage newGray(int height, int width) { return new BufferedImage(width, height, TYPE_BYTE_GRAY); }
    
    public BufferedImage imread(InputStream in) {  try { return ImageIO.read(in); } catch(IOException e) { throw new RuntimeException(e); } }
    public BufferedImage imread(String path) { try { return ImageIO.read(new File(path)); }  catch(IOException e) { throw new RuntimeException(e); } }
    public BufferedImage imread(File file) { try { return ImageIO.read(file); } catch(IOException e) { throw new RuntimeException(e); } }
    public BufferedImage imread(URL url) { try { return ImageIO.read(url); } catch(IOException e) { throw new RuntimeException(e); } }
  
    public void imwrite(BufferedImage img, String format, String path) throws IOException { imwrite(img, format, new File(path)); }
    public void imwrite(BufferedImage img, String format, File file) throws IOException { ImageIO.write(img, format, file); }
    public void imwrite(BufferedImage img, String format, OutputStream out) throws IOException { ImageIO.write(img, format, out); }
    public void imwrite(BufferedImage img, String path) throws IOException { imwrite(img, new File(path)); }
    public void imwrite(BufferedImage img, File file) throws IOException {
        String name = file.getName();
        String format = name.substring(name.lastIndexOf('.') + 1);
        imwrite(img, format, file);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: read HSI raw">
    public byte[] read_raw_bil_dtype12(byte[] bytes) {//[IW, IC, IH]
        if((bytes.length & 1) != 0) throw new IllegalArgumentException("bytes.length % 2 != 0");
        byte[] pixel = new byte[bytes.length >> 1];
        for(int i=0; i<pixel.length; i++) {
            int b0 = bytes[(i << 1)    ] & 0xff;//char -> unsigned char
            int b1 = bytes[(i << 1) + 1] & 0xff;
            pixel[i] = (byte) (((b1 << 8) + b0) >> 4);
        }
        return pixel;
    }
    
    public byte[] read_raw_bil_dtype12(String path) { return read_raw_bil_dtype12(new File(path)); }
    public byte[] read_raw_bil_dtype12(File file) {//[IW, IC, IH]
        byte[] pixel = null; int index = 0;
        FileInputStream in = null;
        BufferedInputStream bfin = null;
        try {
            in = new FileInputStream(file);
            bfin = new BufferedInputStream(in);
                    
            int file_length = (int)file.length();
            if((file_length & 1) != 0) throw new IllegalArgumentException("file_length % 2 != 0");
            pixel = new byte[file_length >> 1];
                
            byte[] buf = new byte[4096];
            int read_size = Math.min(buf.length, 4096), len;
            while((len = bfin.read(buf, 0, read_size)) != -1) {
                for(int i=0; i<len; i+=2) {
                    int b0 = buf[i    ] & 0xff;
                    int b1 = buf[i + 1] & 0xff;
                    pixel[index++] = (byte) (((b1 << 8) + b0) >> 4);
                }
            }
        }
        catch(IOException e) { pixel = null; throw new RuntimeException(e); }
        finally { try {
            if(bfin != null) bfin.close();
            if(in != null) in.close();
        } catch(IOException e) { throw new RuntimeException(e); }}
        return pixel;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: pixelIO">
    public byte[] read_pixels(String path) { return alpha.fl.to_bytes(path); }
    public byte[] read_pixels(File file) { return alpha.fl.to_bytes(file); }
    public void write_pixels(String path, byte[] pixels) { alpha.fl.wt_bytes(path, pixels); }
    public void write_pixels(File file, byte[] pixels) { alpha.fl.wt_bytes(file, pixels); }
    
    public static final int BEST_SPEED = Deflater.BEST_SPEED;
    public static final int BEST_COMPRESSION = Deflater.BEST_COMPRESSION;
    
    private int zip_level = Deflater.BEST_SPEED;
    public int zip_level() { return zip_level; }
    public DragonCV zip_level(int level) { this.zip_level = level; return this; };
    
    private Charset charset = UTF_8;
    public Charset charset() { return charset; }
    public DragonCV charset(Charset cs) { this.charset = cs; return this; }
    
    //<editor-fold defaultstate="collapsed" desc="static class: DimPixel">
    public static final class DimPixel implements Entry<int[], byte[]>{
        public int[] dim;
        public byte[] pixel;
        
        public DimPixel(int[] dim, byte[] pixel) {
            this.dim = dim;
            this.pixel = pixel;
        }
        
        @Override public int[] getKey() { return dim; }
        @Override public byte[] getValue() { return pixel; }
        @Override public byte[] setValue(byte[] value) { byte[] old = pixel; pixel = value; return old; }
        @Override public String toString() { return "{ dim =" + Arrays.toString(dim) + ", length = " + pixel.length + "}"; }
    }
    //</editor-fold>
    
    private final static String zip_dim_key = "dim";
    private final static String zip_pixel_key = "pixel";
    
    //<editor-fold defaultstate="collapsed" desc="inner-code: read_zip_pixel">
    protected byte[] process_zip_pixel(ZipFile zp, ZipEntry entry, int length) {
        InputStream in = null;
        BufferedInputStream bfin = null;
        try {
            in = zp.getInputStream(entry);
            bfin = new BufferedInputStream(in);
            
            byte[] pixel = new byte[length];
            bfin.read(pixel);
            return pixel;
        }
        catch(IOException e) { throw new RuntimeException(e); }
        finally { try {
            if(bfin != null) bfin.close();
            if(in != null) in.close();
        } catch(IOException e) {  throw new RuntimeException(e); } }
    }
    
    protected int[] process_zip_dim(ZipFile zp, ZipEntry entry) {
        InputStream in = null;
        InputStreamReader reader = null;
        BufferedReader bufr = null;
        try {
            in = zp.getInputStream(entry);
            reader = new InputStreamReader(in);
            bufr = new BufferedReader(reader);
            return Vector.to_int_vector(bufr.readLine());
        }
        catch(IOException e) { throw new RuntimeException(e); }
        finally { try {
            if(bufr != null) bufr.close();
            if(reader != null) reader.close();
            if(in != null) in.close();
        } catch(IOException e) { throw new RuntimeException(e); }}
    }
    //</editor-fold>
    
    public DimPixel read_zip_pixels(String path) { return read_zip_pixels(new File(path), null); }
    public DimPixel read_zip_pixels(File file) { return read_zip_pixels(file, null); }
    public DimPixel read_zip_pixels(File file, Charset charset) {
        ZipFile zp = null;
        try {
            if(charset == null) charset = this.charset;
            zp = new ZipFile(file, charset);
            
            ZipEntry dim_entry = null;
            ZipEntry pixel_entry = null;
            Enumeration<? extends ZipEntry> entries = zp.entries(); 
            while(entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();
                if(entry.isDirectory()) continue;
                
                String key = entry.getName();
                if (zip_dim_key.equals(key)) dim_entry = entry;
                else if(zip_pixel_key.equals(key)) pixel_entry = entry;
            }
            
            if(dim_entry == null) throw new IOException("There is no dim entry");
            if(pixel_entry == null) throw new IOException("There is no pixel entry");
            
            int[] dim = process_zip_dim(zp, dim_entry);
            byte[] pixel = process_zip_pixel(zp, pixel_entry, Vector.mul(dim));
            return new DimPixel(dim, pixel);
        }
        catch(IOException e) { throw new RuntimeException(e); }
        finally { try {
            if(zp != null) zp.close();
        } catch(IOException e) { throw new RuntimeException(e); }}
    }
    
    public void write_zip_pixels(String path, byte[] pixels, int... dim) { write_zip_pixels(new File(path), -1, null, pixels, dim); } 
    public void write_zip_pixels(File file, byte[] pixels, int... dim) { write_zip_pixels(file, -1, null, pixels, dim); }
    public void write_zip_pixels(File file, int zip_level, Charset charset, byte[] pixels, int... dim) {
        FileOutputStream fos = null;
        ZipOutputStream zos = null;
        BufferedOutputStream bos = null;
        try {
            fos = new FileOutputStream(file);
            
            if(charset == null) charset = this.charset;
            zos = new ZipOutputStream(fos, charset);
            if(zip_level == -1) zip_level = this.zip_level;
            zos.setLevel(zip_level);
            
            zos.putNextEntry(new ZipEntry(zip_dim_key));
            zos.write(Vector.toString(dim).getBytes());
            zos.putNextEntry(new ZipEntry(zip_pixel_key));
            zos.write(pixels);
        }
        catch(IOException e) { throw new RuntimeException(e); }
        finally { try {
            if (bos != null) bos.close();
            if (zos != null) zos.close();
            if (fos != null) fos.close();
        } catch(IOException e) { throw new RuntimeException(e); }}
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: info of image">
    public int channels(BufferedImage img) { return img.getColorModel().getNumComponents(); }
    public int[] dim(BufferedImage img) { return new int[]{ img.getHeight(), img.getWidth(), channels(img) }; }
    
    public int pixel_size(BufferedImage img) { return img.getColorModel().getPixelSize(); }//for precision and depth of clr
    public int[] channel_pixel_size(BufferedImage img) { return img.getColorModel().getComponentSize(); }
    
    public String brief(BufferedImage img)  {
        StringBuilder sb = new StringBuilder(256);
        sb.append("Image { type = ").append(img.getType()).append(", ");
        sb.append("dim = [");Vector.append(sb, dim(img)); sb.append("], ");
        sb.append("pixelSize = ").append(pixel_size(img)).append(", channelPixelSize = [");
        Vector.append(sb, channel_pixel_size(img)); sb.append("] }");
        return sb.toString();
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: show_frame image">
    protected int[] show_last_yx = null;
    protected int[] show_last_hw = null;
    protected int show_move = 0;
    
    protected void add_mouse_wheel_lisener(JFrame frame, JLabel label, BufferedImage img) {
        if(img.getType() == 0) return;
        frame.addMouseWheelListener((MouseWheelEvent e) -> {
            int notches = e.getWheelRotation();
            float scale = (notches < 0 ? 1.1f : 0.909091f);
            int width  = (int) (frame.getWidth()  * scale);
            int height = (int) (frame.getHeight() * scale);
            frame.setSize(width, height);
            label.setSize(width, height);
            label.setIcon(new ImageIcon(cv.reshape(img, height, width)));
        });
    }
    
    public JFrame imshow(BufferedImage image) { return imshow(image, "Image show"); }
    public JFrame imshow(BufferedImage image, String title) {
        if(image == null) throw new NullPointerException("image is null");

        JFrame frame = new JFrame(title);
        JPanel panel = new JPanel();
        JLabel label = new JLabel();
        
        label.setIcon(new ImageIcon(image));
        panel.add(label);
        frame.add(panel);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.pack();
        
        if(frame.getWidth()  < 256) frame.setSize(256, frame.getHeight());
        if(frame.getHeight() < 256) frame.setSize(frame.getWidth(), 256);
        
        if(show_last_yx == null) { 
            frame.setLocationRelativeTo(null);
            show_last_yx = new int[2];
            show_last_hw = new int[2];
        }
        else {
            Dimension screen_dim = Toolkit.getDefaultToolkit().getScreenSize();
            int sh = screen_dim.height, sw = screen_dim.width;
            int ly = show_last_yx[0], lx = show_last_yx[1];
            int lh = show_last_hw[0], lw = show_last_hw[1];
            
            int moveY = ((show_move & 1) == 0 ? 1 : 0), moveX = 1 - moveY;
            int y = (ly + lh * moveY) % sh;
            int x = (lx + lw * moveX) % sw;
            frame.setLocation(x, y);
        }
        
        show_last_yx[0] = frame.getY(); show_last_hw[0] = frame.getHeight();
        show_last_yx[1] = frame.getX(); show_last_hw[1] = frame.getWidth();
        show_move++;
        
        frame.setVisible(true);
        this.add_mouse_wheel_lisener(frame, label, image);
        return frame;
    }
    
    public JFrame imshow(BufferedImage[][] images) { return imshow(images, "Image show"); }
    public JFrame imshow(BufferedImage[][] images, String title) {
        if(images == null) throw new NullPointerException("images is null");

        JFrame frame = new JFrame(title);
        JPanel panel = new JPanel(); frame.add(panel);
        
        int N = images.length, M = images[0].length;
        panel.setLayout(jui.layout_grid(N, M, 4, 4));
        //panel.setLayout(new FlowLayout(FlowLayout.LEFT, 5, 5));
        
        int[] Hs = new int[M];
        int[] Ws = new int[N];
        for(int i=0; i<N; i++) {
            for(int j=0; j<M; j++) {
                BufferedImage img = images[i][j];
                JLabel label = new JLabel();
                label.setIcon(new ImageIcon(img));
                panel.add(label);
                Hs[j] += img.getHeight();
                Ws[i] += img.getWidth();
            }
        }
       
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setResizable(false);
        frame.pack();
        
        int height = Vector.maxValue(Hs) + 10 * (N + 1);
        int width  = Vector.maxValue(Ws) + 7 * (M + 1);
        frame.setSize(width, height);
        
        if(show_last_yx == null) { 
            frame.setLocationRelativeTo(null);
            show_last_yx = new int[2];
            show_last_hw = new int[2];
        }
        else {
            Dimension screen_dim = Toolkit.getDefaultToolkit().getScreenSize();
            int sh = screen_dim.height, sw = screen_dim.width;
            int ly = show_last_yx[0], lx = show_last_yx[1];
            int lh = show_last_hw[0], lw = show_last_hw[1];
            
            int moveY = ((show_move & 1) == 0 ? 1 : 0), moveX = 1 - moveY;
            int y = (ly + lh * moveY) % sh;
            int x = (lx + lw * moveX) % sw;
            frame.setLocation(x, y);
        }
        
        show_last_yx[0] = frame.getY(); show_last_hw[0] = frame.getHeight();
        show_last_yx[1] = frame.getX(); show_last_hw[1] = frame.getWidth();
        show_move++;
        
        frame.setVisible(true);
        return frame;
    }
    //</editor-fold>    
    
    //<editor-fold defaultstate="collapsed" desc="function: reshape image">    
    //<editor-fold defaultstate="collapsed" desc="function: reshape image">
    public BufferedImage reshape(BufferedImage img, double scaleY, double scaleX) {
        int height = (int) (img.getHeight() * scaleY);
        int width = (int) (img.getWidth() * scaleX);
        return reshape(img, height, width);
    }
    public BufferedImage reshape(BufferedImage img, int height, int width) {
        if (height == img.getHeight() && width == img.getWidth()) return img;
        BufferedImage dst = new BufferedImage(width, height, img.getType());
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width, height, null);
        graph.dispose();
        return dst;
    }
    
    public BufferedImage reshape(BufferedImage img, double scaleY, double scaleX, Color bgColor)  {
        int height = (int) (img.getHeight() * scaleY);
        int width  = (int) (img.getWidth() * scaleX);
        return reshape(img, height, width, bgColor);
    }
    public BufferedImage reshape(BufferedImage img, int height, int width, Color bgColor) {
        if (height == img.getHeight() && width == img.getWidth()) return img;
        BufferedImage dst = new BufferedImage(width, height, img.getType());
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width, height, bgColor, null);
        graph.dispose();
        return dst;
    }
    
    public BufferedImage reshape(BufferedImage img, int height, int width, int type) {
        if (height == img.getHeight() && width == img.getWidth()) return img;
        BufferedImage dst = new BufferedImage(width, height, type);
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width, height, null);
        graph.dispose();
        return dst;
    }
    
    public BufferedImage reshape(BufferedImage img, int height, int width, int type, Color bgColor) {
        if (height == img.getHeight() && width == img.getWidth()) return img;
        BufferedImage dst = new BufferedImage(width, height, type);
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width, height, bgColor, null);
        graph.dispose();
        return dst;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: pad image">
    public BufferedImage pad(BufferedImage img, int ph, int pw) { return pad(img, ph, pw, ph, pw); }
    public BufferedImage pad(BufferedImage img, int ph0, int pw0, int ph1, int pw1) {
        int IH = img.getHeight(), IW = img.getWidth();
        int OH = IH + ph0 + ph1;
        int OW = IW + pw0 + pw1;
        BufferedImage dst = new BufferedImage(OW, OH, img.getType());
        
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, pw0, ph0, IW - 1, IH - 1, 0, 0, IW - 1, IH - 1, null);
        graph.dispose();
        return dst;
    }
    
    public BufferedImage pad(BufferedImage img, int ph, int pw, Color bgColor) { return pad(img, ph, pw, ph, pw, bgColor); }
    public BufferedImage pad(BufferedImage img, int ph0, int pw0, int ph1, int pw1, Color bgColor) {
        int IH = img.getHeight(), IW = img.getWidth();
        int OH = IH + ph0 + ph1;
        int OW = IW + pw0 + pw1;
        BufferedImage dst = new BufferedImage(OW, OH, img.getType());
        
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, pw0, ph0, IW - 1, IH - 1, 0, 0, IW - 1, IH - 1, bgColor, null);
        graph.dispose();
        
        return dst;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: trim image">
    public BufferedImage trim(BufferedImage img, int th, int tw) { return trim(img, th, tw, th, tw); }
    public BufferedImage trim(BufferedImage img, int th0, int tw0, int th1, int tw1) {
        int IH = img.getHeight(), IW = img.getWidth();
        int OH = IH - th0 - th1;
        int OW = IW - tw0 - tw1;
        BufferedImage dst = new BufferedImage(OW, OH, img.getType());
        
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, OW - 1, OH - 1, tw0, th0, IW - tw1, IH - th1, null);
        graph.dispose();
        return dst;
    }
    
     public BufferedImage trim(BufferedImage img, int th, int tw, Color bgColor) { return trim(img, th, tw, th, tw, bgColor); }
    public BufferedImage trim(BufferedImage img, int th0, int tw0, int th1, int tw1, Color bgColor) {
        int IH = img.getHeight(), IW = img.getWidth();
        int OH = IH - th0 - th1;
        int OW = IW - tw0 - tw1;
        BufferedImage dst = new BufferedImage(OW, OH, img.getType());
        
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, OW - 1, OH - 1, tw0, th0, IW - tw1, IH - th1, bgColor, null);
        graph.dispose();
        
        return dst;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="function: crop image">
    public BufferedImage crop(BufferedImage img, 
            double scaleY1, double scaleX1, 
            double scaleY2, double scaleX2) 
    {
        int sh = img.getHeight(), sw = img.getWidth();
        int y1 = (int) (sh*scaleY1), y2 = (int) (sh*scaleY2);
        int x1 = (int) (sw*scaleX1), x2 = (int) (sw*scaleX2);
        return DragonCV.this.crop(img, y1, x1, y2, x2);
    }
    
    public BufferedImage crop(BufferedImage img, 
            double scaleY1, double scaleX1, 
            double scaleY2, double scaleX2,
            Color bgColor) 
    {
        int sh = img.getHeight(), sw = img.getWidth();
        int y1 = (int) (sh*scaleY1), y2 = (int) (sh*scaleY2);
        int x1 = (int) (sw*scaleX1), x2 = (int) (sw*scaleX2);
        return crop(img, y1, x1, y2, x2, bgColor);
    }
    
    public BufferedImage crop(BufferedImage img, int y1, int x1, int y2, int x2) {
        int sh = img.getHeight(), sw = img.getWidth();
        if(x1 >= sw) x1 = sw - 1; if(y1 >= sh) y1 = sh - 1;
        if(x2 >= sw) x2 = sw - 1; if(y2 >= sh) y2 = sh - 1;
        if(x1 > x2) {int t = x1; x1 = x2; x2 = t;}
        if(y1 > y2) {int t = y1; y1 = y2; y2 = t;}
        
        int width = x2 - x1 + 1, height = y2 - y1 + 1;
        BufferedImage dst = new BufferedImage(width, height, img.getType());
        
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width - 1, height - 1, x1, y1, x2, y2, null);
        graph.dispose();
        return dst;
    }
    
    public BufferedImage crop(BufferedImage img, int y1, int x1, int y2, int x2, Color bgColor) {
        int sh = img.getHeight(), sw = img.getWidth();
        if(x1 >= sw) x1 = sw - 1; if(y1 >= sh) y1 = sh - 1;
        if(x2 >= sw) x2 = sw - 1; if(y2 >= sh) y2 = sh - 1;
        if(x1 > x2) {int t = x1; x1 = x2; x2 = t;}
        if(y1 > y2) {int t = y1; y1 = y2; y2 = t;}
        
        int width = x2 - x1 + 1, height = y2 - y1 + 1;
        BufferedImage dst = new BufferedImage(width, height, img.getType());
        
        Graphics2D graph = dst.createGraphics();
        graph.drawImage(img, 0, 0, width - 1, height - 1, x1, y1, x2, y2, bgColor, null);
        graph.dispose();
        
        return dst;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Affine Transform">
    public AffineTransformer affine() { return new AffineTransformer(); }
    
    //<editor-fold defaultstate="collapsed" desc="static class: Affine">
    public static final class AffineTransformer  
    {
        AffineTransform tf = new AffineTransform();
        
        //<editor-fold defaultstate="collapsed" desc="Basic-functions">
        public AffineTransform getTransfrom() {  return tf; }
        public void setTransfrom(AffineTransform transfrom) { this.tf = transfrom; }
        
        public AffineTransformer copy() {
            AffineTransformer copied =  new AffineTransformer();
            copied.tf = (AffineTransform) this.tf.clone();
            return copied;
        }
        
        @Override 
        public int hashCode() { 
            return 95 + Objects.hashCode(this.tf); 
        }

        @Override
        public boolean equals(Object obj)  {
            if(!(obj instanceof AffineTransformer)) return false;
            AffineTransformer af = ( AffineTransformer) obj;
            return af.tf.equals(this.tf);
        }
       
        public void append(StringBuilder sb) {
            double m00 = tf.getScaleX(), m01 = tf.getShearX(), m02 = tf.getTranslateX();
            double m10 = tf.getShearY(), m11 = tf.getScaleY(), m12 = tf.getTranslateY();
            sb.append("Affine Transfomer:");
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
       
        //<editor-fold defaultstate="collapsed" desc="affine operationrs">
        public AffineTransformer toIndentity() {
            tf.setToIdentity();
            return this; 
        }
        
        public AffineTransformer invert() {
            try { tf.invert(); } 
            catch (NoninvertibleTransformException ex) {throw new RuntimeException(ex);}
            return this; 
        }
        
        public AffineTransformer move(int moveY, int moveX) {
            tf.translate(moveX, moveY);
            return this;
        }

        public AffineTransformer scale(double scaleX, double scaleY) {
            tf.scale(scaleX, scaleY); 
            return this;
        }
        
        public AffineTransformer rotate(double theta) {
            tf.rotate(theta);
            return this;
        }
        
        public AffineTransformer rotate(double theta, double anchorY, double anchorX) {
            tf.rotate(theta, anchorX, anchorY);
            return this;
        }
        
        public AffineTransformer angleRotate(double theta) {
            tf.rotate(theta * Math.PI / 180);
            return this;
        }
        
        public AffineTransformer angleRotate(double theta, double anchorY, double anchorX) {
            tf.rotate(theta * Math.PI / 180, anchorY, anchorX);
            return this;
        }

        public AffineTransformer shear(double shearY, double shearX) {
            tf.shear(shearX, shearY); 
            return this;
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="running-area">
        public void draw(BufferedImage src, BufferedImage dst) {
            Graphics2D graph = dst.createGraphics();
            graph.drawImage(src, tf, null);
            graph.dispose();
        }
        
        public BufferedImage transform(BufferedImage src)  {
            int IW = src.getWidth(), IH = src.getHeight();
            double m00 = tf.getScaleX(), m01 = tf.getShearX(), m02 = tf.getTranslateX();
            double m10 = tf.getShearY(), m11 = tf.getScaleY(), m12 = tf.getTranslateY();
                 
            //(0, 0) -> (oh0, ow0), (IH, IW) ->(oh3, ow3)
            float ow0 = (float) m02, ow3 = (float) (m00 * IW + m01 * IH + m02);
            float oh0 = (float) m12, oh3 = (float) (m10 * IW + m11 * IH + m12);

            //(0, IW) -> (oh1, ow1), (IH, 0) -> (oh2, ow2)
            float ow1 = (float) (m00 * IW + m02), ow2 = (float) (m01 * IH + m02);
            float oh1 = (float) (m10 * IW + m12), oh2 = (float) (m11 * IH + m12);

            int OW = (int) (Vector.maxValue(ow0, ow1, ow2, ow3));
            int OH = (int) (Vector.maxValue(oh0, oh1, oh2, oh3));
            return transform(src, OH, OW);
        }
        
        public BufferedImage transform(BufferedImage src, int height, int width)  {
            BufferedImage dst = new BufferedImage(width, height, src.getType());
            Graphics2D graph = dst.createGraphics();
            graph.drawImage(src, tf, null);
            graph.dispose();
            return dst;
        }
        //</editor-fold>
    }
    //</editor-fold>
     //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="ColorSpace Transform">
    private final ColorConvertOp gray_op = new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_GRAY), null);
    private final ColorConvertOp bgr_op = new ColorConvertOp(ColorSpace.getInstance(ColorSpace.CS_sRGB), null);
    
    public BufferedImage to_gray(BufferedImage img) { return gray_op.filter(img, null);  }

    public BufferedImage to_BGR(BufferedImage img) { 
        BufferedImage dst = new BufferedImage(img.getWidth(), img.getHeight(), TYPE_3BYTE_BGR);
        return bgr_op.filter(img, dst); 
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="image & pixel[]">
    public byte[][] pixel_channel_split(byte[] pixel, int... dim) { return pixel_channel_split(pixel, dim[0], dim[1], dim[2]); }
    public byte[][] pixel_channel_split(byte[] pixel, int IH, int IW, int IC) {
        if (pixel.length != IH*IW*IC) throw new IllegalArgumentException(String.format(
                "pixel.length {%d} != IH * IW * IC {%d * %d * %d}",
                pixel.length, IH, IW, IC));
        final int HW = IH * IW;
        byte[][] pix = new byte[IC][HW];
        for (int ic=0; ic<IC; ic++)
        for (int hw=0; hw<HW; hw++)
            pix[ic][hw] = pixel[hw*IC + ic];
        return pix;
    }
    
    public byte[] pixel(BufferedImage img) {
        DataBufferByte dataBuf = (DataBufferByte)img.getRaster().getDataBuffer();
        return dataBuf.getData();
    }
    
    //<editor-fold defaultstate="collapsed" desc="channel4: ARFG, ABGR">
    private static final int[] BYTE4 = {8, 8, 8, 8};
    private static final int[] ARGB = {1, 2, 3, 0};
    private static final int[] ABGR = {3, 2, 1, 0};
    
    protected static void require_ndim4(Tensor X) {
        if(X.ndim() != 4) throw new IllegalArgumentException(String.format(
                "X.ndim { got %d } must == 4", X.ndim()));
    }
    
    public BufferedImage ARGB(Tensor X) { require_ndim4(X); return channel4(X.value_int8(), X.dim(-3), X.dim(-2), ARGB); }
    public BufferedImage ABGR(Tensor X) { require_ndim4(X); return channel4(X.value_int8(), X.dim(-3), X.dim(-2), ABGR); }
    
    public BufferedImage ARGB(byte[] pixel, int height, int width) { return channel4(pixel, height, width, ARGB); }
    public BufferedImage ABGR(byte[] pixel, int height, int width) { return channel4(pixel, height, width, ABGR); }
    
    private BufferedImage channel4(byte[] pixel, int IH, int IW, int[] channel_order) {
        if(pixel == null) throw new NullPointerException("pixel == null");    
        if(pixel.length != IW * IH * 4) throw new IllegalArgumentException(String.format(
                "invalid description for argb images: pixel.length { got %d } != IW { got %d } * IH { got %d } * 4",
                pixel.length, IH, IW));
        
        DataBufferByte buf = new DataBufferByte(pixel, pixel.length);
        ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_sRGB);
        ComponentColorModel colorModel = new ComponentColorModel(cs, BYTE4, true, false, //(hasAlpha = true)
                Transparency.TRANSLUCENT, DataBuffer.TYPE_BYTE);        
        WritableRaster raster = Raster.createInterleavedRaster(buf, IW, IH, IW*4, 4, channel_order, null);
        BufferedImage img = new BufferedImage(colorModel, raster, false, null);
        return img;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="channel3: RGB, BGR">
    private static final int[] BYTE3 = {8, 8, 8};
    private static final int[] RGB = {0, 1, 2};
    private static final int[] BGR = {2, 1, 0};

    protected static void require_ndim3(Tensor X) { if(X.ndim() != 3) throw new IllegalArgumentException(String.format("X.ndim { got %d } must == 3", X.ndim())); }
    protected static void require_ndim3(DimPixel dp) { if(dp.dim[2] != 3) throw new IllegalArgumentException(String.format("X.ndim { got %d } must == 3", dp.dim[2])); }
    
    public BufferedImage RGB(Tensor X) { require_ndim3(X); return channel3(X.value_int8(), X.dim(-3), X.dim(-2), RGB); }
    public BufferedImage BGR(Tensor X) { require_ndim3(X); return channel3(X.value_int8(), X.dim(-3),  X.dim(-2), BGR); }
    
    public BufferedImage RGB(DimPixel dp) { require_ndim3(dp); return channel3(dp.pixel, dp.dim[0], dp.dim[1], RGB); }
    public BufferedImage BGR(DimPixel dp) { require_ndim3(dp); return channel3(dp.pixel, dp.dim[0], dp.dim[1], BGR); }
    
    public BufferedImage RGB(byte[] pixel, int height, int width) { return channel3(pixel, height, width, RGB); }
    public BufferedImage BGR(byte[] pixel, int height, int width) { return channel3(pixel, height, width, BGR); }
    
    private BufferedImage channel3(byte[] pixel, int IH, int IW, int[] channel_order) {
        if(pixel == null) throw new NullPointerException("pixel == null");
        if(pixel.length != IW * IH * 3) throw new IllegalArgumentException(String.format(
                "invalid description for rgb images: pixel.length { got %d } != IH { got %d } * IW { got %d } * 3",
                pixel.length, IH, IW));
      
        DataBufferByte buf = new DataBufferByte(pixel, pixel.length);
        ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_sRGB);
        ComponentColorModel colorModel = new ComponentColorModel(cs, BYTE3, false, false, 
                Transparency.OPAQUE, DataBuffer.TYPE_BYTE);        
        WritableRaster raster = Raster.createInterleavedRaster(buf, IW, IH, IW*3, 3, channel_order, null);
        BufferedImage img = new BufferedImage(colorModel, raster, false, null);
        return img;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="channel1: GRAY">
    private static final int[] BYTE1 = { 8 };
    private static final int[] GRAY = { 0 };
    
    public BufferedImage gray(Tensor X) {
        int ndim = X.ndim(); if(ndim != 2 || ndim != 3) throw new IllegalArgumentException(String.format(
                "X.ndim { got %d } must = { 2 or 3 }", ndim));
        return gray(X.value_int8(), X.dim(1 - ndim), X.dim(-ndim));
    }
    public BufferedImage gray(byte[] pixel, int IH, int IW) {
        if(pixel == null) throw new NullPointerException("pixel == null");
        if(pixel.length != IW * IH) throw new IllegalArgumentException(String.format(
                "invalid description for gray images: pixel.length(%d) != IH { got %d } * IW { got %d }",
                pixel.length, IH, IW));
        
        DataBufferByte buf = new DataBufferByte(pixel, pixel.length);
        ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_GRAY);
        ComponentColorModel colorModel = new ComponentColorModel(cs, BYTE1, false, false, 
                Transparency.OPAQUE, DataBuffer.TYPE_BYTE);        
        WritableRaster raster = Raster.createInterleavedRaster(buf, IW, IH, IW, 1, GRAY, null);
        BufferedImage img = new BufferedImage(colorModel, raster, false, null);
        return img;
    }
    
    public BufferedImage gray(Tensor X, Color black, Color white) {
        int ndim = X.ndim(); if(ndim != 2 || ndim != 3) throw new IllegalArgumentException(String.format(
                "X.ndim { got %d } must = { 2 or 3 }", ndim));
        return gray(X.value_int8(), X.dim(1 - ndim), X.dim(-ndim), black, white);
    }
    public BufferedImage gray(byte[] pixel, int IH, int IW, Color black, Color white) {
        if(pixel == null) throw new NullPointerException("pixel == null");
        if(pixel.length != IW * IH) throw new IllegalArgumentException(String.format(
                 "invalid description for gray images: pixel.length(%d) != IH { got %d } * IW { got %d }",
                pixel.length, IH, IW));
        
        int r0 = black.getRed(), g0 = black.getGreen(), b0 = black.getBlue();
        int r1 = white.getRed(), g1 = white.getGreen(), b1 = white.getBlue();
        
        //Lienar inerpolation between Black and White
        byte[] pix = new byte[pixel.length * 3];
        for (int i=0; i<pixel.length; i++) {
            float v1 = (pixel[i] & 0xff) / 255.0f, v0 = 1.0f - v1; 
            int r = (int) (v1*r1 + v0*r0); if (r > 255) r = 255;//red
            int g = (int) (v1*g1 + v0*g0); if (g > 255) g = 255;//green
            int b = (int) (v1*b1 + v0*b0); if (b > 255) b = 255;//blue
            
            int i3 = i * 3;
            pix[i3    ] = (byte) b;//blue
            pix[i3 + 1] = (byte) g;//green
            pix[i3 + 2] = (byte) r;//red
        } 
        
        DataBufferByte buf = new DataBufferByte(pix, pix.length);
        ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_sRGB);
        ComponentColorModel colorModel = new ComponentColorModel(cs, BYTE3, false, false, 
                Transparency.OPAQUE, DataBuffer.TYPE_BYTE);        
        WritableRaster raster = Raster.createInterleavedRaster(buf, IW, IH, IW*3, 3, BGR, null);
        BufferedImage img = new BufferedImage(colorModel, raster, false, null);
        return img;
    }
    //</editor-fold>
    
    public BufferedImage[] channel_graphics(Tensor X, Color black, Color white) {//HSI[H, W, C]
        return channel_graphics(X.img().pixel(), X.dim(0), X.dim(1), X.dim(2), black, white);
    }
    public BufferedImage[] channel_graphics(byte[] pixel, int IH, int IW, int IC, Color black, Color white) {//HSI[H, W, C]
        BufferedImage[] imgs = new BufferedImage[IC];
        byte[] pix = new byte[IH * IW];//[C, H, W]
        for (int ic=0; ic<IC; ic++) {
            for (int ih=0; ih<IH; ih++) 
            for (int iw=0; iw<IW; iw++) {
                int ihw = ih*IW + iw;
                pix[ihw] = pixel[ihw*IC + ic];
            }
            imgs[ic] = gray(pix, IH, IW, black, white);
        }
        return imgs;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="static-class: Canvas">
    public Canvas canvas(int height, int width) { return new Canvas(newBGR(height, width)); }
    public Canvas canvas(BufferedImage img) { return new Canvas(img); }
    
    public static class Canvas  {
        private final BufferedImage img;
        private final Graphics2D g2d;
        
        public Canvas(BufferedImage img) {
            this.img = img;
            this.g2d = (Graphics2D) img.getGraphics();
        }
        
        public BufferedImage image() { return img; }
        public int height() { return img.getHeight(); }
        public int width() { return img.getWidth(); }
        
        public Color color() { return g2d.getColor(); }
        public Canvas color(Color color) { g2d.setColor(color); return this; }
        
        public Font font() { return g2d.getFont(); }
        public Canvas font(Font font) { g2d.setFont(font); return this; }
        
        //<editor-fold defaultstate="collapsed" desc="fill functions">
        public Canvas fill(Color color) {
           g2d.setColor(color);
           g2d.fillRect(0, 0, img.getWidth(), img.getHeight());
           return this;
        }
        
        public Canvas fill_rect(int y, int x, int H, int W) { g2d.fillRect(x, y, W, H); return this; }
        public Canvas fill_rect(Color c, int y, int x, int H, int W) { g2d.setColor(c); return fill_rect(y, x, H, W); }  

        public Canvas fill_rect3D(int y, int x, int H, int W, boolean raised) { g2d.fill3DRect(x, y, W, H, raised); return this; }
        public Canvas fill_rect3D(Color c, int y, int x, int H, int W, boolean raised) { g2d.setColor(c); return fill_rect3D(y, x, H, W, raised); }

        public Canvas fill_oval(int y, int x, int H, int W) { g2d.fillOval(x, y, W, H); return this; }
        public Canvas fill_oval(Color c, int y, int x, int W, int H) { g2d.setColor(c); return fill_oval(y, x, H, W); }

        public Canvas fill_arc(int y, int x, int H, int W, int start_angle, int arc_angle) {  g2d.fillArc(x, y, W, H, start_angle, arc_angle); return this; }
        public Canvas fill_arc(Color c, int y, int x, int H, int W, int start_angle, int arc_angle) { g2d.setColor(c); return fill_arc(y, x, H, W, start_angle, arc_angle); }

        public Canvas fill_polygon(Color c, int[] ys, int[] xs) { g2d.setColor(c); return fill_polygon(ys, xs); }
        public Canvas fill_polygon(int[] ys, int[] xs) {
            if (xs.length != ys.length) throw new IllegalArgumentException(String.format(
                    "xs.length {got %d} != ys.length {got %d}", xs.length, ys.length));
            g2d.fillPolygon(xs, ys, xs.length);
            return this;
        }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="draw functions">
        public Canvas draw_string(String str, int y, int x) { g2d.drawString(str, x, y); return this; }
        public Canvas draw_string(Color c, Font ft, String str, int y, int x) { g2d.setColor(c); g2d.setFont(ft); return draw_string(str, y, x); }

        public Canvas draw_line(int y0, int x0, int y1, int x1) { g2d.drawLine(x0, y0, x1, y1); return this; }
        public Canvas draw_line(Color c, int y0, int x0, int y1, int x1) { g2d.setColor(c); return draw_line(y0, x0, y1, x1); }

        public Canvas draw_rect(int y, int x, int H, int W) { g2d.drawRect(x, y, W, H); return this; }
        public Canvas draw_rect(Color c, int y, int x, int H, int W) { g2d.setColor(c); return draw_rect(y, x, H, W); }

        public Canvas draw_oval(int y, int x, int H, int W) { g2d.drawOval(x, y, W, H); return this; }
        public Canvas draw_oval(Color c, int y, int x, int H, int W) { g2d.setColor(c); return draw_oval(y, x, H, W); }
        public Canvas draw_point(int y, int x) { return draw_oval(y, x, 1, 1); }
        public Canvas draw_point(Color c, int y, int x) { return draw_oval(c, y, x, 1, 1); }

        public Canvas draw_polygon(Color c, int[] ys, int[] xs) { g2d.setColor(c); return this.draw_polygon(ys, xs); }
        public Canvas draw_polygon(int[] ys, int[] xs) {
           if (xs.length != ys.length) throw new IllegalArgumentException(String.format(
                    "xs.length {got %d} != ys.length {got %d}", xs.length, ys.length));
            g2d.drawPolygon(xs, ys, xs.length);
            return this;
        }
        //</editor-fold>
    }
    //</editor-fold>
}
