/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.factory;

import java.io.InputStream;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiPredicate;
import java.util.function.Predicate;
import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import z.util.ds.tree.ZTreeMap;
import z.util.lang.Lang;
import z.util.xml.exception.UnknownTagException;
import z.util.xml.XML;

/**
 *
 * @author dell
 */
public class Meta extends ZTreeMap<String,Object>
{
    //<editor-fold defaultstate="collapsed" desc="toString-Function">
    public static final BiPredicate NO_ELEMENT_MATRIX = (BiPredicate) (key, value) -> !Lang.is_elem_mat_type(value.getClass());
    public static final BiPredicate NO_ELEMENT_ARRAY = (BiPredicate) (Object key, Object value) -> { Class cls = value.getClass(); return !Lang.is_elem_array_type(cls) && !Lang.is_elem_mat_type(cls); };
    public static final BiPredicate NO_ARRAY = (BiPredicate) (Object key, Object value) -> Lang.is_array(value.getClass())==null;
    
    public String toString(BiPredicate pre) { return toString("\n\t", pre); }
    public String toString(String prefix, BiPredicate pre) {
        StringBuilder sb = new StringBuilder(256).append('{');
        forEach((k, v)-> { if(pre.test(k, v)) 
            sb.append(prefix).append(k).append(" = ").append(Lang.toString(v));
        });
        sb.append('}');
        return sb.toString();
    }
    
    @Override public String toString() { return toString("\n\t"); }
    public String toString(String prefix) {
        StringBuilder sb = new StringBuilder(256).append('{');
        forEach((k, v)-> { 
            sb.append(prefix).append(k).append(" = ").append(Lang.toString(v));
        });
        sb.append('}');
        return sb.toString();
    }
    
    public void toXMLString(StringBuilder sb, String prefix) {
        forEach((k, v)-> {
            sb.append(prefix)
                    .append("<property name=\"").append(k)
                    .append("\" type=\"").append(Lang.class_name(v));
            
            Class cls = v.getClass();
            String vstr = Lang.toString(v, cls);
            if (Lang.is_elem_mat_type(cls)) sb.append("\">").append(vstr).append("</property>\n");
            else sb.append("\" value=\"").append(vstr).append("\"/>\n");
        });
    }
    
    public void toXMLString(StringBuilder sb, String prefix, Predicate<String> pre) {
        forEach((k, v)-> {
            if (!pre.test(k)) return;
            sb.append(prefix)
                    .append("<property name=\"").append(k)
                    .append("\" type=\"").append(Lang.class_name(v));
            
            Class cls = v.getClass();
            String vstr = Lang.toString(v, cls);
            if (Lang.is_elem_mat_type(cls)) sb.append("\">").append(vstr).append("</property>\n");
            else sb.append("\" value=\"").append(vstr).append("\"/>\n");
        });
    }
    
    public String toXMLString(String prefix, Predicate<String> pre) {
        StringBuilder sb = new StringBuilder();
        this.toXMLString(sb, prefix, pre);
        return sb.toString();
    }
    
    public String toXMLString(String prefix) {
        StringBuilder sb = new StringBuilder();
        this.toXMLString(sb, prefix);
        return sb.toString();
    }    
    
    public String toXMLString() {
        StringBuilder sb=new StringBuilder();
        this.toXMLString(sb, "\t");
        return sb.toString();
    }

    public String toXMLStringWithBucket(String prefix, String rootNode) {
        Objects.requireNonNull(rootNode, "Meta.rootNode");
        
        StringBuilder sb=new StringBuilder();
        sb.append("<").append(rootNode).append(">\n");
        this.toXMLString(sb, prefix);
        sb.append("</").append(rootNode).append(">\n");
        return sb.toString();
    }    

    public String toXMLStringWithBucket(String prefix, 
            String rootNode, Predicate<String> pre) {
        Objects.requireNonNull(pre, "Predicate");
        Objects.requireNonNull(rootNode, "Meta.rootNode");
        
        StringBuilder sb=new StringBuilder();
        sb.append("<").append(rootNode).append(">\n");
        this.toXMLString(sb, prefix, pre);
        sb.append("</").append(rootNode).append(">\n");
        return sb.toString();
    }    

    public String toXMLStringWithBucket(String prefix, String doctype, 
            String rootNode) {
        Objects.requireNonNull(doctype, "Meta.Doctype");
        Objects.requireNonNull(rootNode, "Meta.rootNode");
        
        StringBuilder sb = new StringBuilder();
        sb.append("<!DOCTYPE ").append(doctype).append(">\n");
        sb.append("<").append(rootNode).append(">\n");
        this.toXMLString(sb, prefix);
        sb.append("</").append(rootNode).append(">\n");
        return sb.toString();
    }    

    public String toXMLStringWithBucket(String prefix, String doctype, 
            String rootNode, Predicate<String> pre) {
        Objects.requireNonNull(pre, "Predicate");
        Objects.requireNonNull(doctype, "Meta.Doctype");
        Objects.requireNonNull(rootNode, "Meta.rootNode");
        
        StringBuilder sb=new StringBuilder();
        sb.append("<!DOCTYPE ").append(doctype).append(">\n");
        sb.append("<").append(rootNode).append(">\n");
        this.toXMLString(sb, prefix, pre);
        sb.append("</").append(rootNode).append(">");
        return sb.toString();
    }    
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="outer-operators">
    public <T> T getValue(String key) { return (T) get(key); }
    public <T> T getValueOrDefault(String key, Object def) { return (T) getOrDefault(key, def); }
    
    public <T> T putIfExists(String key, Object value) { return (get(key) != null? (T) put(key, value) : null); }
    public <T> T putIf(String k, Object v, BiPredicate pre) { return (pre.test(k, v)? (T)put(k, v) : null); }
    
    public void putAllIfExists(Map<String, ?> map) { map.forEach((k, v)->{ putIfExists(k, v); }); }
    public void putAllIfAbsent(Map<String, ?> map) { map.forEach((k, v)->{ putIfAbsent(k, v); }); }
    
    public void requireNoNull(String key) { if (get(key)==null) throw new NullPointerException(key); }
    public <T> T getValueNoNull(String key) {
        T value = (T) get(key);
        if(value == null) throw new NullPointerException(key);
        return value;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Load-From-Document">
    public Meta load(NodeList prop) throws Exception {
        for(int i=0, len = prop.getLength(); i<len; i++) {
            Node cur = prop.item(i);
            if(cur.getNodeType() != Node.ELEMENT_NODE) continue;
            
            String name = cur.getNodeName();
            if(!name.equals("property")) throw new UnknownTagException(name);
            
            NamedNodeMap attr = XML.getAttributes(cur);
            String key  = XML.getAttributeNoNull(attr, "name");
            String type = XML.getAttributeOrDefault(attr, "type", "String");
            String value = XML.getAttribute(attr, "value");
            if(value == null) value = XML.getFirstChildNodeValue(cur);
            put(key, Lang.convert(value, type));
        }
        return this;
    }
    
    public Meta load(Document dou, String docType, String rootNode) throws Exception {
        XML.checkDocType(dou, docType);
        Node conf = dou.getElementsByTagName(rootNode).item(0);
        return load(conf.getChildNodes());
    }
    
    public Meta load(InputStream in, String docType, String rootNode) throws Exception {
        return load(XML.getDocument(in), docType, rootNode);
    }
    
    public Meta load(String src, String docType, String rootNode) throws Exception {
        return load(XML.getDocument(src), docType, rootNode);
    }
    
    public Meta load(Document dou, String rootNode) throws Exception {
        Node conf = dou.getElementsByTagName(rootNode).item(0);
        return load(conf.getChildNodes());
    }
    
    public Meta load(InputStream in, String rootNode) throws Exception {
        return load(XML.getDocument(in), rootNode);
    }
    
    public Meta load(String src, String rootNode) throws Exception {
        return load(XML.getDocument(src), rootNode);
    }
    
    public Meta load(Document dou) throws Exception {
        Node conf = dou.getElementsByTagName("configuration").item(0);
        return load(conf.getChildNodes());
    }
    
    public Meta load(InputStream in) throws Exception { return load(XML.getDocument(in)); }
    public Meta load(String src) throws Exception { return load(XML.getDocument(src)); }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Builder">
    public static Meta valueOf(Document dou, String docType, String rootNode) throws Exception {
        XML.checkDocType(dou, docType);
        return new Meta().load(dou.getElementsByTagName(rootNode).item(0).getChildNodes());
    }
    public static Meta valueOf(InputStream in, String docType, String rootNode) throws Exception {
        return new Meta().load(in, docType, rootNode);
    }
    public static Meta valueOf(String src, String docType, String rootNode) throws Exception {
        return new Meta().load(src, docType, rootNode);
    }
    
    public static Meta valueOf(Document dou, String rootNode) throws Exception { return new Meta().load(dou, rootNode); }
    public static Meta valueOf(String src, String rootNode) throws Exception { return new Meta().load(src, rootNode); }
    public static Meta valueOf(InputStream in, String rootNode) throws Exception { return new Meta().load(in, rootNode); }
    
    public static Meta valueOf(NodeList prop) throws Exception { return new Meta().load(prop); }
    public static Meta valueOf(Document dou) throws Exception { return new Meta().load(dou); }
    public static Meta valueOf(String src) throws Exception { return new Meta().load(src); }
    public static Meta valueOf(InputStream in) throws Exception { return new Meta().load(in); }
    //</editor-fold>
}
