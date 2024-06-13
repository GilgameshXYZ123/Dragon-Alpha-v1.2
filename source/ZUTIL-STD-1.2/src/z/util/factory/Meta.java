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
import z.util.lang.annotation.Optimizable;
import z.util.xml.exception.UnknownTagException;
import z.util.xml.XML;
import z.util.lang.annotation.Passed;

/**
 *
 * @author dell
 */
public class Meta extends ZTreeMap<String,Object>
{
    //<editor-fold defaultstate="collapsed" desc="toString-Function">
    public static final BiPredicate NO_ELEMENT_MATRIX = new BiPredicate() {
        @Override
        public boolean test(Object key, Object value) 
        {
            return !Lang.isElementMatrixType(value.getClass());
        }
    };
    public static final BiPredicate NO_ELEMENT_ARRAY = new BiPredicate() {
        @Override
        public boolean test(Object key, Object value) 
        {
            Class clazz=value.getClass();
            return !Lang.isElementVectorType(clazz)&&!Lang.isElementMatrixType(clazz);
        }
    };
    public static final BiPredicate NO_ARRAY=new BiPredicate() {
        @Override
        public boolean test(Object key, Object value) 
        {
            return Lang.isArray(value.getClass())==null;
        }
    };
    
    public String toString(String prefix, BiPredicate pre)
    {
        StringBuilder sb=new StringBuilder();
        sb.append('{');
        this.forEach((String key,Object value)->
        {
            if(pre.test(key, value))
                sb.append(prefix).append(key).append(" = ").append(Lang.toString(value));
        });
        sb.append('}');
        return sb.toString();
    }
    public String toString(String prefix)
    {
        StringBuilder sb=new StringBuilder();
        sb.append('{');
        this.forEach((String key,Object value)->
        {
            sb.append(prefix).append(key).append(" = ").append(Lang.toString(value));
        });
        sb.append('}');
        return sb.toString();
    }
    public String toString(BiPredicate pre)
    {
        return toString("\n\t", pre);
    }
    @Override
    public String toString()
    {
        return toString("\n\t");
    }
    
    
    @Optimizable("use Lang.isMatrix to extends Data type")
    @Passed
    public void toXMLString(StringBuilder sb, String prefix)
    {
        this.forEach((String key,Object value)->
        {
            sb.append(prefix).append("<property name=\"").append(key)
              .append("\" type=\"").append(Lang.getClassName(value));
            
            Class clazz=value.getClass();
            String valueStr=Lang.toString(value, clazz);
            if(Lang.isElementMatrixType(clazz)) sb.append("\">").append(valueStr).append("</property>\n");
            else sb.append("\" value=\"").append(valueStr).append("\"/>\n");
        });
    }
    @Passed
    public void toXMLString(StringBuilder sb, String prefix, Predicate<String> pre)
    {
        this.forEach((String key,Object value)->
        {
            if(!pre.test(key)) return;
            sb.append(prefix).append("<property name=\"").append(key)
              .append("\" type=\"").append(Lang.getClassName(value));
            
            Class clazz=value.getClass();
            String valueStr=Lang.toString(value, clazz);
            if(Lang.isElementMatrixType(clazz)) 
            {
                sb.append("\">").append(valueStr).append("</property>\n");
            }
            else sb.append("\" value=\"").append(valueStr).append("\"/>\n");
        });
    }
    public String toXMLString(String prefix, Predicate<String> pre)
    {
        StringBuilder sb=new StringBuilder();
        this.toXMLString(sb, prefix, pre);
        return sb.toString();
    }
    public String toXMLString(String prefix)
    {
        StringBuilder sb=new StringBuilder();
        this.toXMLString(sb, prefix);
        return sb.toString();
    }    
    public String toXMLString()
    {
        StringBuilder sb=new StringBuilder();
        this.toXMLString(sb, "\t");
        return sb.toString();
    }
    @Passed
    public String toXMLStringWithBucket(String prefix, String rootNode)
    {
        Objects.requireNonNull(rootNode, "Meta.rootNode");
        
        StringBuilder sb=new StringBuilder();
        sb.append("<").append(rootNode).append(">\n");
        
        this.toXMLString(sb, prefix);
        
        sb.append("</").append(rootNode).append(">\n");
        return sb.toString();
    }    
    @Passed
    public String toXMLStringWithBucket(String prefix, String rootNode, Predicate<String> pre)
    {
        Objects.requireNonNull(pre, "Predicate");
        Objects.requireNonNull(rootNode, "Meta.rootNode");
        
        StringBuilder sb=new StringBuilder();
        sb.append("<").append(rootNode).append(">\n");
        
        this.toXMLString(sb, prefix, pre);
        
        sb.append("</").append(rootNode).append(">\n");
        return sb.toString();
    }    
      @Passed
    public String toXMLStringWithBucket(String prefix, String doctype, String rootNode)
    {
        Objects.requireNonNull(doctype, "Meta.Doctype");
        Objects.requireNonNull(rootNode, "Meta.rootNode");
        
        StringBuilder sb=new StringBuilder();
        sb.append("<!DOCTYPE ").append(doctype).append(">\n");
        sb.append("<").append(rootNode).append(">\n");
        
        this.toXMLString(sb, prefix);
        
        sb.append("</").append(rootNode).append(">\n");
        return sb.toString();
    }    
    @Passed
    public String toXMLStringWithBucket(String prefix, String doctype, String rootNode, Predicate<String> pre)
    {
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
    
    //<editor-fold defaultstate="collapsed" desc="Outer-Operators">
    public <T> T getValue(String key)
    {
        return (T) this.get(key);
    }
    public <T> T getValueOrDefault(String key, Object def)
    {
        this.getOrDefault(key, def);
        return (T) this.getOrDefault(key, def);
    }
    public <T> T putIfExists(String key, Object value)
    {
        return (this.get(key)!=null? (T)this.put(key, value):null);
    }
    public <T> T putIf(String key, Object value, BiPredicate pre)
    {
        return (pre.test(key, value)? (T)this.put(key, value):null);
    }
    public void putAllIfExists(Map<String,Object> map)
    {
        map.forEach((String key, Object value)->{this.putIfExists(key, value);});
    }
    public void putAllIfAbsent(Map<String,Object> map)
    {
        map.forEach((String key, Object value)->{this.putIfAbsent(key, value);});
    }
    public void requireNoNull(String key)
    {
        if(this.get(key)==null) throw new NullPointerException(key);
    }
    public <T extends Object> T getValueNoNull(String key)
    {
        T value=(T) this.get(key);
        if(value==null) throw new NullPointerException(key);
        return value;
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Load-From-Document">
    @Passed
    public Meta load(NodeList prop) throws Exception {
        Node cur = null;
        String name = null,key = null,value=null,type=null;
        NamedNodeMap attr=null;
        for(int i=0,len=prop.getLength();i<len;i++)
        {
            cur=prop.item(i);
            if(cur.getNodeType()!=Node.ELEMENT_NODE) continue;
            
            name=cur.getNodeName();
            if(!name.equals("property")) throw new UnknownTagException(name);
            
            attr=XML.getAttributes(cur);
            key=XML.getAttributeNoNull(attr, "name");
            type=XML.getAttributeOrDefault(attr, "type", "String");
            value=XML.getAttribute(attr, "value");
            if(value==null) value=XML.getFirstChildNodeValue(cur);
            this.put(key, Lang.convert(value, type));
        }
        return this;
    }
    
    public Meta load(Document dou, String docType, String rootNode) throws Exception
    {
        XML.checkDocType(dou, docType);
        Node conf=dou.getElementsByTagName(rootNode).item(0);
        return this.load(conf.getChildNodes());
    }
    public Meta load(InputStream in, String docType, String rootNode) throws Exception
    {
        return this.load(XML.getDocument(in), docType, rootNode);
    }
    public Meta load(String src, String docType, String rootNode) throws Exception
    {
        return this.load(XML.getDocument(src), docType, rootNode);
    }
    
    public Meta load(Document dou, String rootNode) throws Exception
    {
        Node conf=dou.getElementsByTagName(rootNode).item(0);
        return this.load(conf.getChildNodes());
    }
    public Meta load(InputStream in, String rootNode) throws Exception
    {
        return this.load(XML.getDocument(in), rootNode);
    }
    public Meta load(String src, String rootNode) throws Exception
    {
        return this.load(XML.getDocument(src), rootNode);
    }
    
    public Meta load(Document dou) throws Exception
    {
        Node conf=dou.getElementsByTagName("configuration").item(0);
        return this.load(conf.getChildNodes());
    }
    public Meta load(InputStream in) throws Exception
    {
        return this.load(XML.getDocument(in));
    }
    public Meta load(String src) throws Exception
    {
        return this.load(XML.getDocument(src));
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="Builder">
    public static Meta valueOf(NodeList prop) throws Exception { return new Meta().load(prop); }
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
    
    public static Meta valueOf(Document dou, String rootNode) throws Exception {
        return new Meta().load(dou, rootNode);
    }
    
    public static Meta valueOf(String src, String rootNode) throws Exception {
        return new Meta().load(src, rootNode);
    }
    
    public static Meta valueOf(InputStream in, String rootNode) throws Exception {
        return new Meta().load(in, rootNode);
    }
    
    public static Meta valueOf(Document dou) throws Exception {
        return new Meta().load(dou);
    }
    public static Meta valueOf(String src) throws Exception {
        return new Meta().load(src);
    }
    public static Meta valueOf(InputStream in) throws Exception {
        return new Meta().load(in);
    }
    //</editor-fold>
}
