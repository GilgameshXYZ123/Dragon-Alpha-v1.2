/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.xml;

import z.util.xml.exception.TagMissedException;
import z.util.xml.exception.DuplicatedTagException;
import z.util.xml.exception.DocTypeMismatchedException;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.DocumentType;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;
import z.util.lang.Resources;
import z.util.xml.exception.NullAttributeException;
import z.util.xml.exception.NullNodeException;
import z.util.lang.annotation.Passed;

/**
 *
 * @author dell
 */
public class XML
{
    //<editor-fold defaultstate="collapsed" desc="Init-Code">
    private static DocumentBuilderFactory factory = null;
    private static DocumentBuilder builder = null;
    
    static {
        synchronized(XML.class) {
            try {
                if(factory == null) factory = DocumentBuilderFactory.newInstance();
                if(builder == null) builder = factory.newDocumentBuilder();
            } catch(ParserConfigurationException e) { e.printStackTrace(); }
        }
    }
    //</editor-fold>
    
    //<editor-fold defaultstate="collapsed" desc="XML-DocumentGetter">
    public static Document getDocument(InputStream in) throws Exception { return builder.parse(in); }
    public static Document getDocument(String url)  {
        Document dou = null;
        InputStream in = null;
        try {
            in = Resources.getResourceAsStream(url);
            dou = builder.parse(in);
        }
        catch(IOException | SAXException e) { throw new RuntimeException(e); }
        finally { try{ 
            if(in != null) in.close();
        } catch(IOException e) { throw new RuntimeException(e); } }
        return dou;
    }
    
    public static Document getDocument(byte[] buffer) {
        Document dou = null;
        InputStream in = null;
        try {
            in = new ByteArrayInputStream(buffer);
            dou = builder.parse(in);
        } 
        catch(IOException | SAXException e) { throw new RuntimeException(e); }
         finally { try{ 
            if(in != null) in.close();
        } catch(IOException e) { throw new RuntimeException(e); } }
        return dou;
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="XML-Getter">
    public static NamedNodeMap getAttributes(Node n) {
        NamedNodeMap attr = n.getAttributes();
        if(attr == null) throw new NullAttributeException(n);
        return attr;
    }
    
    /**
     * @param attributes
     * @param name
     * @return null if no such attribute
     */
    public static String getAttribute(NamedNodeMap attributes, String name) {
        Node n = attributes.getNamedItem(name);
        if(n == null) return null;
        String value = n.getNodeValue();
        return value;
    }
    /**
     * @param attributes
     * @param name
     * @throws NullNodeException if the specified item of attributes is null
     * @throws NullPointerException if the value of the specific item is null
     * @return 
     */
    @Passed
    public static String getAttributeNoNull(NamedNodeMap attributes, String name)
    {
        Node n=attributes.getNamedItem(name);
        if(n==null) throw new NullNodeException(name);
        String value=n.getNodeValue();
        if(value==null) throw new NullPointerException("Node["+n.getNodeName()+"]."+value);
        return value;
    }
    /**
     * @param attributes
     * @param name
     * @param DEF_VALUE
     * @return DEF_VALUE if the specific item of the item.value is null
     */
    @Passed
    public static String getAttributeOrDefault(NamedNodeMap attributes, String name, String DEF_VALUE)
    {
        Node n=attributes.getNamedItem(name);
        if(n==null) return DEF_VALUE;
        String value=n.getNodeValue();
        return value==null? DEF_VALUE:value;
    }
    @Passed
    public static String getFirstChildNodeValue(Node n)
    {
        Node fn=n.getFirstChild();
        if(fn==null) throw new NullNodeException(n, ".firstChildNode");
        return fn.getNodeValue();
    }
    @Passed
    public static Node getFirstElemenChildtNode(Node n)
    {
        NodeList child=n.getChildNodes();
        Node cur=null;
        for(int i=0,len=child.getLength();i<len;i++)
        {
            cur=child.item(i);
            if(cur.getNodeType()==Node.ELEMENT_NODE) return cur;
        }
        throw new NullNodeException("Node["+n.getNodeName()+"] has no child ElementNode");
    }
    @Passed
    public static Node getFirstTextChildNode(Node n)
    {
        NodeList child=n.getChildNodes();
        Node cur=null;
        for(int i=0,len=child.getLength();i<len;i++)
        {
            cur=child.item(i);
            if(cur.getNodeType()==Node.TEXT_NODE) return cur;
        }
        throw new NullNodeException("Node["+n.getNodeName()+"] has no child TextNode");
    }
    @Passed
    public static Node getFirstCommentChildNode(Node n)
    {
        NodeList child=n.getChildNodes();
        Node cur=null;
        for(int i=0,len=child.getLength();i<len;i++)
        {
            cur=child.item(i);
            if(cur.getNodeType()==Node.COMMENT_NODE) return cur;
        }
        throw new NullNodeException("Node["+n.getNodeName()+"] has no child CommentNode");
    }
    @Passed
    public static Node getFirstCDataChildNode(Node n)
    {
        NodeList child=n.getChildNodes();
        Node cur=null;
        for(int i=0,len=child.getLength();i<len;i++)
        {
            cur=child.item(i);
            if(cur.getNodeType()==Node.CDATA_SECTION_NODE) return cur;
        }
        throw new NullNodeException("Node["+n.getNodeName()+"] has no child CDataNode");
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="XML-Checker">
    /**
     * if doctype is null, then check the Document Type, else return;
     * @param dou
     * @param doctype 
     * @throws NullPointerException if dou.DocType is null
     * @throws NullPointerException if dou.DocType.name is null
     */
    public static void checkDocType(Document dou, String doctype) {
        if (doctype == null) return;
        DocumentType type = dou.getDoctype();
        if (type == null) throw new NullPointerException("Doctype is null");
        String name = type.getName();
        if (name == null) throw new NullPointerException("Doctype.name is null");
        if (!name.equalsIgnoreCase(doctype)) throw new DocTypeMismatchedException();
    }
    
    public static void requireNoDuplicate(Node cur) { if(cur != null) throw new DuplicatedTagException(); }
    public static void requireNoMissed(Node cur) { if(cur == null) throw new TagMissedException(); }
    //</editor-fold>
}
