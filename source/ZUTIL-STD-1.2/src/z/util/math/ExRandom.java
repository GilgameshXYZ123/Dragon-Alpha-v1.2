/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.math;
import java.lang.reflect.Array;
import java.util.Objects;
import java.util.Random;
import z.util.lang.annotation.Passed;
import z.util.math.vector.Vector;
import java.util.function.IntFunction;
        
/**
 *
 * @author dell
 */
public class ExRandom extends Random
{
    private static final long serialVersionUID = 12233124124L;
    
    private static final String NEGATIVE_WIDTH = "Matrix.height<=0";
    private static final String NEGATIVE_HEIGHT = "Matrix.width<=0";
    private static final String NO_RANDOM_SPACE = "max==min:";
    
    public ExRandom() {} 
    public ExRandom(long seed) { super(seed); }

    //<editor-fold defaultstate="collapsed" desc="Array-Checker">
    private static void checkIntMatrix(int height, int width, long min, long max) {
        if(height<=0) throw new BadBoundException(NEGATIVE_HEIGHT);
        if(width<=0) throw new BadBoundException(NEGATIVE_WIDTH);
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
   
    private static void checkIntMatrix(int[][] v, int min, int max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }

    private static void checkIntMatrix(long[][] v, long min, long max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }

    @Passed
    private static void checkRealMatrix(int height, int width, double min, double max)
    {
        if(height<=0) throw new BadBoundException(NEGATIVE_HEIGHT);
        if(width<=0) throw new BadBoundException(NEGATIVE_WIDTH);
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
  
    @Passed
    private static void checkRealMatrix(float[][] v, float min, float max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
  
    @Passed
    private static void checkRealMatrix(double[][] v, double min, double max)
    {
        if(v==null) throw new NullPointerException();
        if(min==max) throw new BadBoundException(NO_RANDOM_SPACE);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="class: BadBoundException">
    public static class BadBoundException extends RuntimeException 
    {
        public static final String MSG = "BadBoundException:";
        public BadBoundException() {}
        public BadBoundException(String message) { super(MSG + message); }
        public BadBoundException(Throwable cause) { super(cause); }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="extensive: next_number">
    //<editor-fold defaultstate="collapsed" desc="number: long">
    protected long __next_long(long bound) {
        long r = ((long)next(31) << 32) + next(32);
        return r % bound;
    }
    
    public long nextLong(long max) {
        if(max == 0) return 0;
        if(max < 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must >= 0", max));
        long bound = max + 1;
        if(bound > Long.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must < Long.MAX_VALUE { got %d }",
                max, Long.MAX_VALUE));
        return __next_long(bound);
    }
    
    public long nextLong(long min, long max) {
        if(min == max) return min;
        if(min > max) { long t = min; min = max; max = t; }
        long bound = max - min + 1;
        if(bound > Long.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "(max { got %d } - min { got %d }) must < Integer.MAX_VALUE { got %d }", 
                max, min, Long.MAX_VALUE));
        return __next_long(bound) + min;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="number: int">
    @Override
    public int nextInt(int max) {
        if(max == 0) return 0;
        int bound = max + 1;
        if(bound > Integer.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must < Integer.MAX_VALUE { got %d }",
                max, Integer.MAX_VALUE));
        long v = (long) next(32) & 0xffffffffL;
        return (int) (v % bound);
    }
    
    public int nextInt(int min, int max) {
        if(min == max) return min;
        if(min > max) { int t = min; min = max; max = t; }
        long bound = max - min + 1;
        long v = (long) next(32) & 0xffffffffL;
        return (int) (v % bound + min);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="number: short">
    protected short next_short = 0;
    protected boolean have_next_short = false;
    
    public synchronized short nextShort() {
        if(have_next_short) { have_next_short = false; return next_short; }
        int r = next(32);
        have_next_short = true;
        next_short = (short) r;
        return       (short) (r >> 16);
    }
    
    public short nextShort(short max) {
        if(max == 0) return 0;
        int bound = max + 1;
        if(bound > Short.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must < Long.MAX_VALUE { got %d }",
                max, Short.MAX_VALUE));
        int r = (nextShort() & 0xffff);
        return (short) (r % bound);
    }
    
    public short nextShort(short min, short max) {
        if(min == max) return min;
        if(min > max) { short t = min; min = max; max = t; }
        int bound = max - min + 1;
        int r = (nextShort() & 0xffff);
        return (short) (r % bound + min);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="number: char">
    protected char next_char = 0;
    protected boolean have_next_char = false;
    
    public synchronized char nextChar() {
        if(have_next_char) { have_next_char = false; return next_char;}
        int r = next(32);
        have_next_char = true;
        next_char = (char) r;
        return      (char) (r >> 16);
    }
    
    public char nextChar(char max) {
        if(max == 0) return 0;
        int bound = max + 1;
        if(bound > Character.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must < Long.MAX_VALUE { got %d }",
                (int)max, (int)Character.MAX_VALUE));
        int r = (nextChar() & 0xffff);
        return (char) (r % bound);
    }
    
    public char nextChar(char min, char max) {
        if(min == max) return min;
        if(min > max) { char t = min; min = max; max = t; }
        int bound = max - min + 1;
        int r = (nextChar() & 0xffff);
        return (char) (r % bound + min);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="number: byte">
    protected byte next_byte = 0;
    protected boolean have_next_byte = false;
    
    public synchronized byte nextByte() {
        if(have_next_byte) { have_next_byte = false; return next_byte; }
        short r = nextShort();
        have_next_byte = true;
        next_byte = (byte) r;
        return      (byte) (r >> 8);
    }
    
    public byte nextByte(byte max) {
        if(max == 0) return 0;
        int bound = max + 1;
        if(bound > Byte.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must < Long.MAX_VALUE { got %d }",
                max, Byte.MAX_VALUE));
        int r = (nextByte() & 0xff);
        return (byte) (r % bound);
    }
    
    public byte nextByte(byte min, byte max) {
        if(min == max) return min;
        if(min > max) { byte t = min; min = max; max = t; }
        int bound = max - min + 1;
        int r = (nextByte() & 0xff);
        return (byte) (r % bound + min);
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="number: double">
    public double nextDouble(double min, double max) { return (max - min) * nextDouble() + min;  }
    public double nextDouble(double max) {
        if(max <= 0.0) throw new IllegalArgumentException(String.format(
                "max { got %f } must > 0", max));
        return max * nextDouble(); 
    }
    
    public double nextGaussian(double sigma) {
        if(sigma <= 0.0) throw new IllegalArgumentException(String.format(
                "sigma { got %f } must > 0", sigma));
        return sigma * nextGaussian();
    }
    
    public double nextGaussian(double mu, double sigma) { 
        if(sigma <= 0.0) throw new IllegalArgumentException(String.format(
                "sigma { got %f } must > 0", sigma));
        return nextGaussian() * sigma + mu; 
    } 
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="number: float">
    public float nextFloat(float max) {
        if(max <= 0.0f) throw new IllegalArgumentException(String.format(
                "max { got %f } must > 0", max));
        if(max > Float.MAX_VALUE) throw new IllegalArgumentException(String.format(
                "max { got %f } must < Float.MAX_VALUE { got %f }", 
                max, Float.MAX_VALUE));
        return max * nextFloat();
    }
     
    public float nextFloat(float min, float max) {  
        if(max == min) return min;
        if(max < min) { float t = max; max = min; min = t; }
        double bound = (double)max - min;
        return (float) (bound * nextFloat() + min);  
    }
    
    protected boolean have_next_gaussianf = false;
    protected float next_gaussianf = 0.0f;
    protected final double TWO_PI = 2.0 * Math.PI;
    
    public synchronized float nextGaussianf() { 
        if(have_next_gaussianf) { have_next_gaussianf = false; return next_gaussianf; }

        double v1 = nextFloat(); if(v1 <= 0.0f) v1 = 1e-8f;//v1 != 0
        double v2 = nextFloat() * TWO_PI;//v2 = v2 * 2 * PI
        v1 = Math.sqrt(-2.0 * Math.log(v1));//v1 = sqrt(-2 * log(v1))
        
        have_next_gaussianf  = true;
        next_gaussianf = (float) (v1 * Math.sin(v2));//r1 = v1 * cos(v2)
        return           (float) (v1 * Math.cos(v2));//r2 = v1 * sin(v2)
    }
    
    public float nextGaussianf(float sigma) {
        if(sigma <= 0.0f) throw new IllegalArgumentException(String.format(
                "max { got %f } must > 0", sigma));
        return sigma * nextGaussianf();
    }
    
    public float nextGaussianf(float mu, float sigma) {
        if(sigma <= 0.0f) throw new IllegalArgumentException(String.format(
                "sigma { got %f } must > 0", sigma));
        return sigma * nextGaussianf() + mu; 
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Vector: random">
    //<editor-fold defaultstate="collapsed" desc="vector: double">
    public double[] next_double_vector(int length) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        
        double[] arr = new double[length];
        for(int i=0; i<length; i++) arr[i] = nextDouble();
        return arr;
    }
 
    public double[] next_double_vector(int length, double max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == 0.0) return new double[length];
        if(max < 0.0) throw new IllegalArgumentException(String.format(
                "max { got %f } must >= 0", max));
        
        double[] arr = new double[length];
        for(int i=0; i<length; i++) arr[i] = nextDouble() * max;
        return arr;
    }
    
    public double[] next_double_vector(int length, double min ,double max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == min) return Vector.constants(max, length);
        
        double threshold = max - min;
        double[] arr = new double[length];
        for(int i=0; i<length; i++) arr[i] = nextDouble() * threshold + min;
        return arr;
    }
  
    public double[] next_gaussian_vector(int length) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        
        double[] arr=new double[length];
        for(int i=0;i<length;i++) arr[i]=this.nextGaussian();
        return arr;
    }
    
    public double[] next_gaussian_vector(int length, double max) { 
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == 0.0) return new double[length];
        if(max < 0.0) throw new IllegalArgumentException(String.format(
                "max { got %f } must >= 0", max));
        
        double[] arr = new double[length];
        for(int i=0; i<length; i++) arr[i] = this.nextGaussian() * max;
        return arr;
    }
    
    public double[] next_gaussian_vector(int length, double min, double max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == min) return Vector.constants(min, length);
        
        double threshold = max - min;
        double[] arr = new double[length];
        for(int i=0; i<length; i++) arr[i] = this.nextGaussian()*threshold + min;
        return arr;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="vector: float">
    public float[] next_float_vector(int length) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        float[] arr = new float[length];
        for(int i=0; i<length; i++) arr[i] = nextFloat();
        return arr;
    }
    
    public float[] next_float_vector(int length, float max) { 
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == 0.0f) return new float[length];
        if(max < 0.0f) throw new IllegalArgumentException(String.format(
                "max { got %f } must >= 0", max));
        if(max > Float.MAX_VALUE) throw new IllegalArgumentException(String.format(
                "max { got %f } must < Float.MAX_VALUE { got %f }", 
                max, Float.MAX_VALUE));
        
        float[] arr = new float[length];
        for(int i=0; i<length; i++) arr[i] = nextFloat() * max;
        return arr;
    }
    
    public float[] next_float_vector(int length, float min, float max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == min) return Vector.constants(min, length);
        
        if(max < min) { float t = max; max = min; min = t; }
        double bound = (double)max - min;
        
        float[] arr = new float[length];
        for(int i=0; i<length; i++) arr[i] = (float) (nextFloat() * bound + min);
        return arr;
    }
    
    public float[] next_gaussianf_vector(int length) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        
        float[] arr = new float[length];
        for(int i=0;i<length;i++) arr[i] = nextGaussianf();
        return arr;
    }
    
    public float[] next_gaussianf_vector(int length, float sigma) { 
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(sigma <= 0.0f) throw new IllegalArgumentException(String.format(
                "sigma { got %f } must > 0", sigma));
        
        float[] arr = new float[length];
        for(int i=0; i<length; i++) arr[i] = nextGaussianf() * sigma;
        return arr;
    }
    
    public float[] next_gaussianf_vector(int length, float mu, float sigma) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(sigma <= 0.0f) throw new IllegalArgumentException(String.format(
                "sigma { got %f } must > 0", sigma));
        
        float[] arr = new float[length];
        for(int i=0; i<length; i++) arr[i] = nextGaussianf() * sigma + mu;
        return arr;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="vector: long">
    public long[] next_long_vector(int length) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        
        long[] arr = new long[length];
        for(int i=0; i<length;i++) arr[i] = nextLong();
        return arr;
    }
    
    public long[] next_long_vector(int length, long max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == 0) return new long[length];
        if(max < 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must > 0", length));
        
        long bound = max + 1;
        if(bound > Long.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must < Long.MAX_VALUE { got %d }",
                max, Long.MAX_VALUE));
            
        long[] arr = new long[length];
        for(int i=0; i<length; i++) arr[i] = __next_long(bound);
        return arr;
    }
    
    public long[] next_long_vector(int length, long min, long max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(min == max) return Vector.constants(max, length);
        
        if(max < min) { long t = max; max = min; min = t; }
        long bound = max - min + 1;
        if(bound > Long.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "(max { got %d } - min { got %d }) must < Integer.MAX_VALUE { got %d }", 
                max, min, Long.MAX_VALUE));
        
        long[] arr = new long[length];
        for(int i=0; i<length; i++) arr[i] = __next_long(bound) + min;
        return arr;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapased" desc="vector: int">
    public int[] next_int_vector(int length) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        
        int[] arr = new int[length];
        for(int i=0; i<length; i++) arr[i] = next(32);
        return arr;
    }

    public int[] next_int_vector(int length, int max) { 
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == 0) return new int[length];
        
        int bound = max + 1;
        if(bound > Integer.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must < Integer.MAX_VALUE { got %d }",
                max, Integer.MAX_VALUE));
        
        int[] arr = new int[length];
        for(int i=0; i<length; i++) {
            long v = (long) next(32) & 0xffffffffL;
            arr[i] =  (int) (v % bound);
        }
        return arr;
    }
    
    public int[] next_int_vector(int length, int min ,int max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == min) return Vector.constants(min, length);
        
        if(max < min) { int t = max; max = min; min = t; }
        int bound = max - min + 1;
      
        int[] arr = new int[length];
        for(int i=0; i<length; i++) {
            long v = (long) next(32) & 0xffffffffL;
            arr[i] = (int) (v % bound + min);
        }
        return arr;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapased" desc="vector: short">
    public void nextShorts(short[] shorts) {
        int len2 = shorts.length >> 1, rlen = shorts.length & 1;
        int index = 0;
        for(int i=0; i<len2; i++) {
            int r = next(32);
            shorts[index++] = (short) r;
            shorts[index++] = (short) (r >> 16);
        }
        if(rlen != 0) { shorts[index++] = (short) next(32); }
    }
    
    public short[] next_short_vector(int length) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        short[] arr = new short[length]; nextShorts(arr);
        return arr;
    }
    
    public short[] next_short_vector(int length, short max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == 0) return new short[length];
        if(max <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must >= 0", max));
        
        int bound = max + 1; 
        if(bound > Short.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must < Integer.MAX_VALUE { got %d }",
                max, Short.MAX_VALUE));
        
        short[] arr = new short[length]; nextShorts(arr);
        
        for(int i=0; i<arr.length; i++) {
            int v = arr[i] & 0xffff;//signed -> unsigned
            arr[i] = (short) (v % bound);
        }
        return arr;
    }
    
    public short[] next_short_vector(int length, short min, short max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(min == max) return Vector.constants(max, length);
        
        if(max < min) { short t = min; min = max; max = t; }
        int bound = max - min + 1;
        short[] arr = new short[length]; nextShorts(arr);
        
        for(int i=0; i<arr.length; i++) {
            int v = arr[i] & 0xffff;//signed ->  unsigned
            arr[i] = (short) (v % bound + min);
        }
        return arr;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapased" desc="vector: char">
    public void nextChars(char[] chars) {
        int len2 = chars.length >> 1, rlen = chars.length & 1;//in32 / in16 = 2
        int index = 0;
        for(int i=0; i<len2; i++) {
            int r = next(32);
            chars[index++] = (char) r;
            chars[index++] = (char) (r >> 16);
        }
        if(rlen != 0) { chars[index++] = (char) next(32); }//rlen = 1
    }
    
    public char[] next_char_vector(int length) { 
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        char[] arr = new char[length]; nextChars(arr);
        return arr;
    }
    
    public char[] next_char_vector(int length, char max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == 0) return new char[length];
        if(max < 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must >= 0", (int)max));
       
        int bound = max + 1;
        if(bound > Character.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must < Integer.MAX_VALUE { got %d }",
                (int)max, (int)Character.MAX_VALUE));
        
        char[] arr = new char[length]; nextChars(arr);
        for(int i=0; i<arr.length; i++) {
            int v = arr[i] & 0xffff;//signed -> unsinged
            arr[i] = (char) (v % bound);
        }
        return arr;
    }
    
    public char[] next_char_vector(int length, char min, char max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(min == max) return Vector.constants(min, length);
        
        if(max < min) { char t = min; min = max; max = t; }
        int bound = max - min + 1;
        
        char[] arr = new char[length]; nextChars(arr);
        for(int i=0; i<arr.length; i++) {
            int v = arr[i] & 0xffff;//signed -> unsigned
            arr[i] = (char) (v % bound + min);
        }
        return arr;
    }
    
    public String nextString(int length) { return new String(next_char_vector(length)); }
    public String nextString(int length, char min, char max) { return new String(next_char_vector(length, min, max)); }
    //</editor-fold>
    //<editor-fold defaultstate="collapased" desc="vector: byte">
    @Override
    public void nextBytes(byte[] bytes) {
        int len4 = bytes.length >> 2;//in32 / in8 = 4
        int rlen = bytes.length & 3;
                
        int index = 0;
        for(int i=0; i < len4; i++) {
            int r = next(32);
            bytes[index++] = (byte) r;
            bytes[index++] = (byte) (r >> 8);
            bytes[index++] = (byte) (r >> 16);
            bytes[index++] = (byte) (r >> 24);
        }
        
        if(rlen != 0) {//reln = { 1, 2, 3}
            int r = next(32);
            bytes[index++] = (byte) r;
            if(rlen > 1) bytes[index++] = (byte) (r >> 8);
            if(rlen > 2) bytes[index++] = (byte) (r >> 16);
        }
    }
   
    public byte[] next_byte_vector(int length) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        byte[] arr = new byte[length]; nextBytes(arr);
        return arr;
    }
    
    public byte[] next_byte_vector(int length, byte max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must >= 0", max));
        
        int bound = max + 1;
        if(bound > Byte.MAX_VALUE || bound <= 0) throw new IllegalArgumentException(String.format(
                "max { got %d } must < Integer.MAX_VALUE { got %d }",
                max, Byte.MAX_VALUE));
        
        byte[] arr = new byte[length]; nextBytes(arr);
        for(int i=0; i<arr.length; i++) {
            int v = arr[i] & 0xff;//signed -> unsigned
            arr[i] = (byte) (v % bound);
        }
        return arr;
    }
    
    public byte[] next_byte_vector(int length, byte min, byte max) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        if(max == min) return Vector.constants(min, length);
        if(max < min) { byte t = min; min = max; max = t; }
        int bound = max - min + 1;
        byte[] arr = new byte[length]; nextBytes(arr);
        
        for(int i=0; i<arr.length; i++) {
            int v = arr[i] & 0xff;//signed -> unsigned
            arr[i] = (byte) (v % bound + min);
        }
        return arr;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="vector: object">
    public <T> T[] next_object_vector(int length, Class<T> clazz, IntFunction<T> func) {
        if(length <= 0) throw new IllegalArgumentException(String.format(
                "length { got %d } must > 0", length));
        
        T[] arr = (T[]) Array.newInstance(clazz, length);
        for(int i=0; i<length; i++) arr[i] = func.apply(nextInt());
        return arr;
    }
    //</editor-fold>
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix: random">
    //<editor-fold defaultstate="collapsed" desc="matrix: float">
    public float[][] next_float_mat(int height, int width) {
        if(height <= 0) throw new IllegalArgumentException(String.format(
                "height { got %d } must > 0", height));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } must > 0", width));
        
        float[][] arr = new float[height][width];
        for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++) arr[i][j] = nextFloat();
        return arr;
    }

    public float[][] next_float_mat(int height, int width, float max) {
        if(height <= 0) throw new IllegalArgumentException(String.format(
                "height { got %d } must > 0", height));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } must > 0", width));
        if(max == 0.0f) return new float[height][width];
        if(max > Float.MAX_VALUE) throw new IllegalArgumentException(String.format(
                "max { got %f } must < Float.MAX_VALUE { got %f }", 
                max, Float.MAX_VALUE));
        
        float[][] arr = new float[height][width];
        for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++) arr[i][j] = nextFloat() * max;
        return arr;
    }
    
    public float[][] next_float_mat(int height, int width, float min, float max) {
        if(height <= 0) throw new IllegalArgumentException(String.format(
                "height { got %d } must > 0", height));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } must > 0", width));
        if(max < min) { float t = max; max = min; min = t; }
        double bound = (double)max - min;
        
        float[][] arr = new float[height][width];
        for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++) arr[i][j] = (float) (nextFloat() * bound + min);
        return arr;
    }
    
    public float[][] next_gaussianf_mat(int height, int width) {
        if(height <= 0) throw new IllegalArgumentException(String.format(
                "height { got %d } must > 0", height));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } must > 0", width));
        
        float[][] arr = new float[height][width];
        for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++) arr[i][j] = nextGaussianf();
        return arr;
    }
    
    public float[][] next_gaussianf_mat(int height, int width, float sigma) {
        if(height <= 0) throw new IllegalArgumentException(String.format(
                "height { got %d } must > 0", height));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } must > 0", width));
        if(sigma <= 0.0f) throw new IllegalArgumentException(String.format(
                "sigma { got %f } must > 0", sigma));
        
        float[][] arr = new float[height][width];
        for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++) arr[i][j] = nextGaussianf() * sigma;
        return arr;
    }
    
    public float[][] next_gaussianf_mat(int height, int width, float mu, float sigma) {
        if(height <= 0) throw new IllegalArgumentException(String.format(
                "height { got %d } must > 0", height));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } must > 0", width));
        if(sigma <= 0.0f) throw new IllegalArgumentException(String.format(
                "sigma { got %f } must > 0", sigma));
        
        float[][] arr = new float[height][width];
        for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++) arr[i][j] = nextGaussianf() * sigma + mu;
        return arr;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="matrix: double">
    public double[][] next_double_matrix(int height, int width) {
        if(height <= 0) throw new IllegalArgumentException(String.format(
                "height { got %d } must > 0", height));
        if(width <= 0) throw new IllegalArgumentException(String.format(
                "width { got %d } must > 0", width));
        
        double[][] arr = new double[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextDouble();
        return arr;
    }
    
    public double[][] nextGaussianMatrix(int height, int width)
    {
        double[][] arr=new double[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextGaussian();
        return arr;
    }
   
    public double[][] nextDoubleMatrix(int height, int width, double max)
    {
        return this.nextDoubleMatrix(height, width, 0, max);
    }
    @Passed
    public double[][] nextGaussianMatrix(int height, int width, double max)
    {
        return this.nextDoubleMatrix(height, width, 0, max);
    }
    @Passed
    public double[][] nextDoubleMatrix(double[][] v, double max)
    {
        this.nextDoubleMatrix(v, 0, max);
        return v;
    }
    @Passed
    public double[][] nextGaussianMatrix(double[][] v, double max)
    {
        this.nextGaussianMatrix(v, 0, max);
        return v;
    }
    
    public double[][] nextDoubleMatrix(int height, int width, double max, double min) {
        ExRandom.checkRealMatrix(height, width, min, max);
        double[][] arr=new double[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextDouble()*max+min;
        return arr;
    }

    public double[][] nextGaussianMatrix(int height, int width, double max, double min) {
        ExRandom.checkRealMatrix(height, width, min, max);
        double[][] arr = new double[height][width];
        for(int i=0; i<height; i++)
        for(int j=0; j<width; j++) arr[i][j]=this.nextGaussian()*max+min;
        return arr;
    }
    @Passed
    public double[][] nextDoubleMatrix(double[][] v, double max, double min)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkRealMatrix(v, min, max);
        double threshold=max-min;
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextDouble()*threshold+min;
        }
        return v;
    }
    @Passed
    public double[][] nextGaussianMatrix(double[][] v, double max, double min)
    {
        if(max<min) {double t=max;max=min;min=t;}
        ExRandom.checkRealMatrix(v, min, max);
        double threshold=max-min;
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextGaussian()*threshold+min;
        }
        return v;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix:exRandom:int">
    @Passed
    public int[][] nextIntMatrix(int height, int width)
    {
        int[][] arr=new int[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextInt();
        return arr;
    }
    @Passed
    public int[][] nextIntMatrix(int[][] v)
    {
        Objects.requireNonNull(v);
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextInt();
        }
        return v;
    }
    @Passed
    public int[][] nextIntMatrix(int height, int width, int max)
    {
        return this.nextIntMatrix(height, width, 0, max);
    }
    @Passed
    public int[][] nextIntMatrix(int[][] v, int max)
    {
        this.nextIntMatrix(v, 0, max);
        return v;
    }
    @Passed
    public int[][] nextIntMatrix(int height, int width, int min, int max)
    {
        if(max<min) {int t=max;max=min;min=t;}
        ExRandom.checkIntMatrix(height, width, min, max);
        int threshold=max-min;
        int[][] arr=new int[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextInt(threshold)+min;
        return arr;
    }
    @Passed
    public int[][] nextIntMatrix(int[][] v, int min, int max)
    {
        if(max<min) {int t=max;max=min;min=t;}
        ExRandom.checkIntMatrix(v, min, max);
        int threshold=max-min;
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextInt(threshold)+min;
        }
        return v;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Matrix:exRandom:long">
    @Passed
    public long[][] nextLongMatrix(int height, int width)
    {
        long[][] arr=new long[height][width];
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=this.nextLong();
        return arr;
    }
    @Passed
    public void nextLongMatrix(long[][] v)
    {
        Objects.requireNonNull(v);
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;j++) v[i][j]=this.nextInt();
        }
    }
    @Passed
    public long[][] nextLongMatrix(int height, int width, long max)
    {   
        return this.nextLongMatrix(height, width, 0, max);
    }
    @Passed
    public void nextLongMatrix(long[][] v, long max)
    {
        this.nextLongMatrix(v, 0, max);
    }
    @Passed
    public long[][] nextLongMatrix(int width, int height, long min, long max)
    {
        if(max<min) {long t=max;max=min;min=t;}
        ExRandom.checkIntMatrix(height, width, min, max);
        long[][] arr=new long[height][width];
        long threshold=max-min;
        for(int i=0,j;i<height;i++)
        for(j=0;j<width;j++) arr[i][j]=(long) (this.nextDouble()*threshold+min);
        return arr;
    }
    @Passed
    public void nextLongMatrix(long[][] v, long min ,long max)
    {
        if(max<min) {long t=max;max=min;min=t;}
        ExRandom.checkIntMatrix(v, min, max);
        long threshold=max-min;
        for(int i=0,j;i<v.length;i++)
        {
            Objects.requireNonNull(v[i]);
            for(j=0;j<v[i].length;i++) v[i][j]=(long) (this.nextDouble()*threshold+min);
        }
    }
    //</editor-fold>
    //</editor-fold>
}
