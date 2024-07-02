/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common;

import java.awt.Color;
import java.awt.Font;
import java.util.HashMap;
import z.dragon.common.DragonCV.Canvas;
import static z.dragon.common.DragonCV.cv;
import static z.ui.JUI.FontHandler.font;
import z.util.lang.Lang;
import z.util.math.ExRandom;
import z.util.math.vector.Vector;
import z.util.math.vector.Vector.MaxMin;
import static z.ui.JUI.ColorHandler.clr;

/**
 *
 * @author Gilgamesh
 */
public final class DragonPlot 
{
    private DragonPlot() {}
    
    public static final DragonPlot plot = new DragonPlot();
    
    //<editor-fold defaultstate="collapsed" desc="class: Axis">
    public static class Axis {
        private String name;//tap of axis
        private float start, end;
        private float[] scales;//relative index from [0-1]
        private String[] taps;
        private Font tapFt = font.Times_New_Roman(12);
        private Font nameFt = font.italic(font.Times_New_Roman(16));
        
        public Axis(float[] scales, String[] taps, float start, float end) {
            this.name = "Axis";
            if (scales.length !=  taps.length) throw new IllegalArgumentException(String.format(
                    "scales.length {got %d} != taps.length {got %d}", 
                    scales.length, taps.length));
            this.start = start; 
            this.end = end;
            this.scales = scales;
            this.taps = taps;
        }
        
        //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
        public String name() { return name; }
        public Axis name(String name) { this.name = name; return this; }
        
        public float start() { return start; }
        public float end() { return end; }
        
        public float[] scales () { return scales; }
        public Axis scales(float[] scales) { this.scales = scales; return this; }
        
        public String[] taps() { return taps; }
        public Axis taps(String[] taps) { this.taps = taps; return this; }
        
        public Font tap_font() { return tapFt; }
        public Axis tap_font(Font ft) { this.tapFt = ft; return this; }
        
        public Font name_font() { return nameFt; }
        public Axis name_font(Font ft) { this.nameFt = ft; return this; } 
        //</editor-fold>
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="axis: creators">
    public Axis axis(float[] scales, String[] taps, float start, float end) { return new Axis(scales, taps, start, end); }
    
    public Axis axis(float e) { return axis(0, e); }
    public Axis axis(float s, float e) { return axis(s, e, (e - s) / 8); }
    public Axis axis(float s, float e, float stride) { //[start][start + scale * i][end]
        if(s > e) { float t = s; s = e; e = t; }
        float div = e - s; 
        int n = (int) Math.ceil(div / stride);
        float[] scales = new float[n + 1];
        String[] taps = new String[n + 1];
        scales[0] = 0; taps[0] = Float.toString(s); 
        scales[n] = 1; taps[n] = Float.toString(e);
        for(int i=1; i<n; i++) {//n + 1 point, 0, n
            float v = s + stride * i;
            scales[i] = v / div; 
            taps[i] = Float.toString(v);
        }
        return new Axis(scales, taps, s, e);
    }
    
    public Axis axis(int e) { return axis(0, e); }
    public Axis axis(int s, int e) { return axis(s, e, (e - s) / 8); }
    public Axis axis(int s, int e, int stride) { //[start][start + scale * i][end]
        if(s > e) { int t = s; s = e; e = t; }
        int div = e - s;
        int n = (div + stride - 1) / stride;//ceiling
        float[] scales = new float[n + 1];
        String[] taps  = new String[n + 1];
        scales[0] = 0; taps[0] = Integer.toString(s); 
        scales[n] = 1; taps[n] = Integer.toString(e);
        for(int i=1; i<n; i++) {
            int v = s + stride * i;
            scales[i] = 1.0f * v / div; 
            taps[i] = Integer.toString(v);
        }
        return new Axis(scales, taps, s, e);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: Line">
    public static class Line {
        private String name;//name of column
        private Color c = clr.blue();//color of line
        private float[] ys;//indices on yaxis
        private float[] xs;//indices on xaxis
        private Font nameFt = font.italic(font.Times_New_Roman(18));
        
        public Line(String name, float[] ys, float[] xs) {
            this.yxs(ys, xs);
            this.name = name;
        }
        
        //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
        public String name() { return name; }
        public Line name(String name) { this.name = name; return this; }
        
        public Color color() { return c; }
        public Line color(Color c) { this.c = c; return this; }
        
        public Font name_font() { return nameFt; }
        public Line name_font(Font ft) { nameFt = ft; return this; }
        
        public int number() { return xs.length; }
        public float[] ys() { return ys; }
        public float[] xs() { return xs; }
        public Line yxs(float[] ys, float[] xs) {
            if (ys.length != xs.length) throw new IllegalArgumentException(String.format(
                    "ys.length {got %d} != xs.length{got %d}", 
                    ys.length, xs.length));
            this.ys = ys;
            this.xs = xs;
            return this;
        }
        //</editor-fold>
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="line: creators">
    public Line line(float[] ys, float[] xs) { return new Line(null, ys, xs); }
    public Line line(float[] ys) {
        float[] xs = new float[ys.length]; 
        float stride = 1.0f / xs.length;
        for(int i=0; i<ys.length; i++) xs[i] = stride * i;
        return new Line(null, ys, xs);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: LineChart">
    public static class LineChart  {
        private String name;
        private Axis yaxis, xaxis;
        private final HashMap<String, Line> map = new HashMap<>();
        private int my = 1, mx = 1;//margin = m * tap_font.size
        private Font nameFt = font.bold_italic(font.Times_New_Roman(20));
        
        public LineChart(Axis yaxis, Axis xaxis, Line... lines) {
            this.yaxis = yaxis;
            this.xaxis = xaxis;
            for (Line line : lines) map.put(line.name, line);
        }
        
        //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
        public Axis yaxis() { return yaxis; }
        public LineChart yaxis(Axis yaxis) { this.yaxis = yaxis; return this; }
        
        public Axis xaxis() { return xaxis; }
        public LineChart xaxis(Axis xaxis) { this.xaxis = xaxis; return this; }
        
        public HashMap<String, Line> lines() { return map; }
        
        public String name() { return name; }
        public LineChart name(String name) { this.name = name; return this; } 
        
        public Font name_font() { return nameFt; }
        public LineChart name_font(Font ft) { this.nameFt = ft; return this; }
        //</editor-fold>
        //<editor-fold defaultstate="collapsed" desc="draw functions">
        protected void adjust_myx() {
            String[] xtap = xaxis.taps;
            my = 1;
            for (String tap : xtap) { int len = (tap.length()); if(mx < len) mx = len; }
        }
        
        protected int marginX() { return (int) (mx * xaxis.tapFt.getSize() * 1.1f); }
        protected int marginY() { return (int) (my * yaxis.tapFt.getSize() * 1.1f); }
        protected int marginT() { return nameFt.getSize(); }
        
        protected void draw_name(Canvas ca) {
            if(name == null || name.isEmpty()) return;
            int W = ca.width(), MT = marginT();
            int tX = (int)(W - name.length() * nameFt.getSize()*0.5f) >> 1;
            ca.draw_string(clr.ink_black(), nameFt, name, MT, tX);
        }
        
        protected void draw_yaxis(Canvas ca) {
            int H = ca.height();
            int MY = marginY(), MX = marginX(), MT = marginT();
            
            int xs = MX;
            int ys = MY + MT, ye = H - MY, LH = ye - ys;
            int yas = ys - ((MY * 3) >> 2);//end of arrow: [ys, ye]->[yas]
            
            ca.draw_line(clr.black(), yas, xs, ye, xs);
            ca.draw_line(yas + 5, xs - 5, yas, xs);//draw arrow
            ca.draw_line(yas + 5, xs + 5, yas, xs);//draw arrow
            ca.font(yaxis.nameFt);//name font
            ca.draw_string(yaxis.name, ys, xs + yaxis.nameFt.getSize());
            
            ca.font(yaxis.tapFt);//tap font
            float[] scs = yaxis.scales; String[] taps = yaxis.taps; 
            for(int i=0; i<scs.length; i++) {
                int y = ys + (int) (scs[scs.length - 1 - i] * LH);
                if (i < scs.length - 1) ca.draw_line(y, xs, y, xs + 5);
                
                String tap = taps[i];
                int tx = xs - MX; if (tx < 0) tx = 0;
                int ty = y + (MY >> 1); if (ty > H) ty = H;
                ca.draw_string(tap, ty, tx);
            }
        }
        
        protected void draw_xaxis(Canvas ca) {
            int H = ca.height(), W = ca.width();
            int MY = marginY(), MX = marginX();
            int ys = H - MY;
            int xs = MX, xe = W - MX, LW = xe - xs;
            int xea = xe + ((MX * 3) >> 2);//end of arrow: [xs, xe]->[ae]
            
            ca.draw_line(clr.black(), ys, xs, ys, xea);
            ca.draw_line(ys - 5, xea - 5, ys, xea);//draw arrow
            ca.draw_line(ys + 5, xea - 5, ys, xea);//draw arrow
            ca.font(xaxis.nameFt);//name font
            ca.draw_string(xaxis.name, 
                    ys - xaxis.nameFt.getSize(), 
                    xea - xaxis.nameFt.getSize() * xaxis.name.length() / 2);
           
            ca.font(xaxis.tapFt);//tap font 
            float[] scs = xaxis.scales; String[] taps = xaxis.taps; 
            for(int i=0; i<scs.length; i++) {
                int x = xs + (int) (scs[i] * LW);
                if (i > 0) ca.draw_line(ys - 5, x, ys, x);
                
                String tap = taps[i];
                int ty = ys + MY; if (ty > H) ty = H;
                int tx = x - (int)(tap.length() * 0.5f) * xaxis.tapFt.getSize(); if (tx < 0) tx = 0;
                ca.draw_string(tap, ty, tx);
            }
        }
        
        protected void draw_line_name(Canvas ca, Line line, int tx, int index) {
           if (line.name != null) {
                int MT = marginT();
                int ty = MT + (index + 1)*line.nameFt.getSize() - 5;
                ca.draw_string(line.color(), line.nameFt, line.name, ty, tx);
                ca.fill_rect((int) (ty - line.nameFt.getSize()*0.5f), tx - 20, 5, 15);
            }
        }
        
        protected void draw_line(Canvas ca, Line line) {
            int H = ca.height(), W = ca.width();
            int MY = marginY(), MX = marginX(), MT = marginT();
            
            int LH = H - 2*MY - MT;//MY + LH = H
            int LW = W - 2*MX;//MX + LW = W
            int ys = H - MY;
            
            float ystart = yaxis.start(), div = yaxis.end() - ystart;
            ca.color(line.color());
            for(int i=1; i<line.ys.length; i++) {
                float y0 = line.ys[i - 1], y1 = line.ys[i];
                float x0 = line.xs[i - 1], x1 = line.xs[i];
                
                float ly0 = (y0 - ystart) / div;
                float ly1 = (y1 - ystart) / div;
                
                int Y0 = ys - (int)(ly0 * LH), X0 = MX + (int)(x0 * LW);
                int Y1 = ys - (int)(ly1 * LH), X1 = MX + (int)(x1 * LW);
                
                ca.draw_line(Y0, X0, Y1, X1);
            }
        }
        
        public Canvas draw(int H, int W) {
            Canvas ca = cv.canvas(H, W);
            adjust_myx();
            ca.fill(Color.white);
            draw_name(ca);
            draw_xaxis(ca);
            draw_yaxis(ca);
            
            map.values().forEach((line) -> { draw_line(ca, line); });
            int mintx = W;//idx for line name
            for (Line line : map.values()) {
                if (line.name == null || line.name.isEmpty()) continue;
                int tx = (int) (W - line.nameFt.getSize()* line.name.length()*0.5f - 10);
                if (mintx > tx) mintx = tx;
            }
            int idx = 0; 
            for (Line line : map.values()) this.draw_line_name(ca, line, mintx, idx++);
            return ca;
        }
        //</editor-fold>
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="line chart: creators">
    public LineChart line_chart(Axis yaxis, Axis xaxis, Line... lines) { return new LineChart(yaxis, xaxis, lines); }
    
    public LineChart line_chart(Axis yaxis, Line... lines) { 
        int maxlen = lines[0].xs.length;
        for(int i=1; i<lines.length; i++) {
            Line line = lines[i];
            if(maxlen < line.xs.length) maxlen = line.xs.length;
        }
        return new LineChart(yaxis, axis(maxlen), lines);
    }
    
    public LineChart line_chart(Line... lines) { 
        MaxMin<Float> mm = Vector.maxMin(lines[0].ys);
        float ymax = mm.max(), ymin = mm.min();
        int maxlen = lines[0].xs.length;
        for(int i=1; i<lines.length; i++) {
            Line line = lines[i];
            MaxMin<Float> tmm = Vector.maxMin(line.ys);
            float tymax = tmm.max(); if(tymax > ymax) ymax = tymax;
            float tymin = tmm.min(); if(tymin < ymin) ymin = tymin;
            if(maxlen < line.xs.length) maxlen = line.xs.length;
        }
        return new LineChart(axis(ymin, ymax), axis(maxlen), lines);
    }
    //</editor-fold>
}
