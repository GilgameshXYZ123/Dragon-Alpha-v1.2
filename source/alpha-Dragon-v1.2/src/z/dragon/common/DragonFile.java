/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.dragon.common;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.function.Consumer;
import java.util.function.Function;
import z.util.math.vector.Vector;

/**
 *
 * @author Gilgamesh
 */
public class DragonFile {
    private DragonFile() {}
    
    public static final DragonFile instance() { return fl; }
    public static final DragonFile fl = new DragonFile();
    
    public BufferedFile create(File file) { return new BufferedFile(file); }
    public BufferedFile create(String path) { return new BufferedFile(new File(path)); }
    
    public byte[] to_bytes(BufferedFile bf) { return bf.to_bytes(); }
    public byte[] to_bytes(File file) { return create(file).to_bytes(); }
    public byte[] to_bytes(String path) { return create(path).to_bytes(); }
    public void wt_bytes(BufferedFile buf, byte... bytes) { buf.wt_bytes(bytes); }
    public void wt_bytes(File file, byte... bytes) { create(file).wt_bytes(bytes); }
    public void wt_bytes(String path, byte... bytes) { create(path).wt_bytes(bytes); }
    
    public char[] to_chars(BufferedFile file) { return file.to_chars(); }
    public char[] to_chars(File file) { return create(file).to_chars(); }
    public char[] to_chars(String path) { return create(path).to_chars(); }
    public void wt_chars(BufferedFile buf, char... chars) { buf.wt_chars(chars); }
    public void wt_chars(File file, char... chars) { create(file).wt_chars(chars); }
    public void wt_chars(String path, char... chars) { create(path).wt_chars(chars); }
    
    public int[] to_int_array(String path) { return create(path).to_int_array(); }
    public void wt_int_array(String path, int... arr) { create(path).wt_int_array(arr); }
    
    public float[] to_float_array(String path) { return create(path).to_float_array(); }
    public void wt_float_array(String path, float... arr) { create(path).wt_float_array(arr); }
    
    public double[] to_double_array(String path) { return create(path).to_double_array(); }
    public void wt_double_array(String path, double... arr) { create(path).wt_double_array(arr); }
    
    public void for_each_line(BufferedFile bf, Consumer<String> con) { bf.for_each_line(con); }
    public void for_each_line(File file, Consumer<String> con) { create(file).for_each_line(con); }
    public void for_each_line(String path, Consumer<String> con) { create(path).for_each_line(con); }
    
    public BufferedFile copy(BufferedFile src, File dst) { return src.copy(dst); }
    public BufferedFile copy(File src, File dst) { return create(src).copy(dst); }
    public BufferedFile copy(String src, String dst) { return create(src).copy(new File(dst)); }
    
    public BufferedFile move(BufferedFile src, File dst) { return src.move(dst); }
    public BufferedFile move(File src, File dst) { return create(src).move(dst); }
    public BufferedFile move(String src, String dst) { return create(src).move(new File(dst)); }
    
    //<editor-fold defaultstate="collapsed" desc="static class: BufferedFile">
    public static class BufferedFile 
    {
        protected final File file;
        protected int buf_size = 2048;
        
        public BufferedFile(File file) { this.file = file; }
        
        //<editor-fold defaultstate="collapsed" desc="Basic-Functions">
        public File file() { return file; }
        
        public void append(StringBuilder sb) {
            sb.append(getClass().getSimpleName()).append("{ ");
            sb.append(file).append(" }");
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder(128);
            this.append(sb);
            return sb.toString();
        }

        @Override
        protected void finalize() throws Throwable {
            super.finalize();
            if(reader != null)  reader.close(); 
            if(bufReader != null) bufReader.close();
            if(writer != null) writer.close();
            if(bufWriter != null) bufWriter.close();
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="IO: bytes">
        public byte[] to_bytes() {
            byte[] buf = null;
            FileInputStream in = null;
            BufferedInputStream bfin = null;
            try {
                in = new FileInputStream(file);
                bfin = new BufferedInputStream(in);
                buf = new byte[check_file_length()];
                
                int index = 0, len;
                int read_size = Math.min(buf.length, buf_size);
                while((len = bfin.read(buf, index, read_size)) != -1 && read_size != 0) {
                    index += len;
                    read_size = buf.length - index;
                    if(read_size > buf_size) read_size = buf_size;
                }
            }
            catch(IOException e) { buf = null; throw new RuntimeException(e); }
            finally { try { 
                if(bfin != null) bfin.close();
                if(in != null) in.close();
            } catch(IOException e) { throw new RuntimeException(e); } }
            return buf;
        }
        
        public void wt_bytes(byte... bytes) {
            FileOutputStream out = null;
            BufferedOutputStream bout = null;
            try {
                out = new FileOutputStream(file);
                bout = new BufferedOutputStream(out);
                bout.write(bytes);
            }
            catch(IOException e) { throw new RuntimeException(e); }
            finally{ try {
                if(bout != null) bout.close();
                if(out != null) out.close();
            } catch(IOException e) { throw new RuntimeException(e); } }
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="IO: chars">
        public char[] to_chars() {
            char[] buf = null;
            FileReader fr = null;
            BufferedReader bufr = null;
            try {
                fr = new FileReader(file);
                bufr = new BufferedReader(fr);
                 
                buf = new char[check_file_length()];
                int index = 0, len;
                int read_size = Math.min(buf.length, buf_size);
                while((len = bufr.read(buf, index, read_size)) != -1 && read_size != 0) {
                    index += len;
                    read_size = buf.length - index;
                    if(read_size > buf_size) read_size = buf_size;
                }
            }
            catch(IOException e) { buf = null; throw new RuntimeException(e); }
            finally { try {
                if(bufr != null) bufr.close();
                if(fr != null) fr.close();
            }catch(IOException e) { throw new RuntimeException(e); }}
            return buf;
        }
        
        public void wt_chars(char[] chars) {
            FileWriter bw = null;
            BufferedWriter bfw = null;
            try {
                bw = new FileWriter(file);
                bfw = new BufferedWriter(bw);
                bfw.write(chars);
            }
            catch(IOException e) { throw new RuntimeException(e); }
            finally{ try {
                if(bfw != null) bfw.close();
                if(bw != null) bw.close();
            } catch(IOException e) { throw new RuntimeException(e); } }
        }
        //</editor-fold>
        
        public void wt_int_array(int... arr) { wt_bytes(Vector.toString(arr).getBytes()); }
        public int[] to_int_array() { return Vector.to_int_vector(new String(to_bytes())); }
        
        public void wt_float_array(float... arr) { wt_bytes(Vector.toString(arr).getBytes()); }
        public float[] to_float_array() { return Vector.to_float_vector(new String(to_bytes())); }
        
        public void wt_double_array(double... arr) { wt_bytes(Vector.toString(arr).getBytes()); }
        public double[] to_double_array() { return Vector.to_double_vector(new String(to_bytes())); }
        
        protected FileReader reader;
        protected BufferedReader bufReader;
        //<editor-fold defaultstate="collapsed" desc="file read">
        protected final void read_close() throws IOException {
            if(bufReader != null) { bufReader.close(); bufReader = null; }
            if(reader != null) { reader.close(); reader = null;}
        }

        protected final void read_open() throws IOException {
            try {
                reader = new FileReader(file);
                bufReader = new BufferedReader(reader);
            }
            catch(FileNotFoundException e) { read_close(); throw e; }
        }
        
        protected final int check_file_length()  {
            if(file.length() >= Integer.MAX_VALUE) throw new IllegalArgumentException(String.format(
                    "file.length { got %d } exceeds the maximum array length { got %d }",
                    file.length(), Integer.MAX_VALUE));
            return (int) file.length();
        }
        
        public String nextLine() throws IOException {
            if (bufReader == null) read_open();
            String line = bufReader.readLine();
            if (line == null) read_close();
            return line;
        }
        
        public ArrayList<String> to_lines() { return to_lines(null); }
        public ArrayList<String> to_lines(Function<String, String> filter) {
            final Function<String, String> ft = (filter == null ? Function.identity() : filter);
            ArrayList<String> arr = new ArrayList<>(1024);
            this.for_each_line((String line) -> { arr.add(ft.apply(line));});
            return arr;
        }
        
        public BufferedFile for_each_line(Consumer<String> con) {
            FileReader fr = null;
            BufferedReader bufr = null;
            try
            {
                fr = new FileReader(file);
                bufr = new BufferedReader(fr);
                for(String line; (line = bufr.readLine()) != null;) con.accept(line);
            }
            catch(IOException e) { throw new RuntimeException(e); }
            finally {  try {
                if(fr != null) fr.close();
                if(bufr != null) bufr.close();
            } catch(IOException e) { throw new RuntimeException(e); }}
            return this;
        }
        //</editor-fold>
        
        protected FileOutputStream writer;
        protected BufferedOutputStream bufWriter;
        //<editor-fold defaultstate="collapsed" desc="file write">
        private void writeClose() throws IOException {
            if(bufWriter != null) bufWriter.close();
            if(writer != null) writer.close();
        }
        
        private void writeOpen() throws IOException {
            try {
                writer = new FileOutputStream(file);
                bufWriter = new BufferedOutputStream(writer);
            }
            catch(IOException e) {
                writeClose(); throw e;
            }
        }
        
        public BufferedFile write(byte[] buf) throws IOException {
            if(bufWriter == null) writeOpen();
            bufWriter.write(buf);
            return this;
        }
        
        public BufferedFile write(String buf) throws IOException {
            if(bufWriter == null) writeOpen();
            bufWriter.write(buf.getBytes());
            return this;
        }
        
        public BufferedFile flush() throws IOException {
            if(bufWriter != null) bufWriter.flush();
            return this;
        }
        
        public BufferedFile finish() throws IOException {
            writeClose();
            return this;
        }
        //</editor-fold>
        
        //<editor-fold defaultstate="collapsed" desc="extra: operations">
        public void delete() { file.delete(); }

        public BufferedFile move(File dst) { 
            try { Files.move(file.toPath(), dst.toPath()); }
            catch (IOException e) { throw new RuntimeException(e); }
            return new BufferedFile(dst);
        }
        
        public BufferedFile copy(File dst) {
            try { Files.copy(file.toPath(), dst.toPath()); }
            catch (IOException e) { throw new RuntimeException(e); }
            return new BufferedFile(dst);
        }
        //</editor-fold>
    }
    //</editor-fold>
}
