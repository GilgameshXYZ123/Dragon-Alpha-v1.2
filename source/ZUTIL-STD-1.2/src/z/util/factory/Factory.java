/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package z.util.factory;

import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.util.function.Consumer;

import org.w3c.dom.Document;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import z.util.ds.tree.ZTreeMap;

import z.util.factory.imp.Factoriable;
import z.util.factory.imp.Initable;
import z.util.xml.exception.TagMissedException;
import z.util.xml.exception.UnknownTagException;
import z.util.xml.XML;
import z.util.lang.annotation.Passed;
/**
 *
 * @author dell
 */
public class Factory implements Factoriable<Document, String, Initable>
{
    ZTreeMap<String,NameSpace> nss=new ZTreeMap<>();
    
    //<editor-fold defaultstate="collapsed" desc="Basic-Function">
    @Override
    public String toString()
    {
        StringBuilder sb=new StringBuilder();
        sb.append("\n\n=============================Factory==================================\n");
        nss.forEach((String key, NameSpace value)-> {sb.append(value).append('\n');});
        sb.append("\n=============================Factory==================================\n\n");
        return sb.toString();
    }
    public String toDetailString()
    {
        StringBuilder sb=new StringBuilder();
        sb.append("\n\n=============================Factory==================================\n");
        nss.forEach((String key, NameSpace value)-> 
        {
            sb.append("-------------------------------\n");
            sb.append(value.toDetailString()).append('\n');
            sb.append("-------------------------------\n\n");
        });
        sb.append("\n=============================Factory==================================\n\n");
        return sb.toString();
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Operators">
    public Resource getResource(String namespace, String name)
    {
        NameSpace ns=nss.get(namespace);
        if(ns==null) throw new NullPointerException("No such NameSpace:"+namespace);
        Resource rs=ns.get(name);
        return rs;
    }
    public NameSpace getNameSpace(String namespace)
    {
        NameSpace ns=nss.get(namespace);
        if(ns==null) throw new NullPointerException("No such NameSpace:"+namespace);
        return ns;
    }
    @Override
    public void removeNameSpace(String name) 
    {
        NameSpace ns=this.nss.remove(name);
        if(ns==null) System.err.println("No such Namespace: "+name);
        else System.out.println("NameSpace is removed: "+ns);
    }
    @Override
    public void clearNameSpace()
    {
        if(nss.isEmpty()) System.err.println("No Namespace to remove");
        else
        {
            nss.clear();
            System.out.println("All NameSpaces have been removed");
        }
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Resource-Loader">
    public static final String DOCTYPE_RESOURCE="Resource";
    
    @Passed
    public void load(Document dou, boolean slient) throws Exception
    {
        //check dock type-------------------------------------------------------
        XML.checkDocType(dou, DOCTYPE_RESOURCE);
        
        //load namespace--------------------------------------------------------
        Node conf=dou.getElementsByTagName("configuration").item(0);
        if(conf==null) throw new TagMissedException("configuration");
        
        NodeList child=conf.getChildNodes();
        String name=null;
        Node cur=null;
        NameSpace ns,last;
        for(int i=0,len=child.getLength();i<len;i++)
        {
            cur=child.item(i);
            if(cur.getNodeType()!=Node.ELEMENT_NODE) continue;
            name=cur.getNodeName();
            if(!name.equals("namespace")) throw new UnknownTagException(name);
            
            ns=NameSpace.valueOf(cur);
            last=nss.put(ns.name, ns);
            if(last!=null) ns.merge(last);
            
            if(!slient)System.out.println("NameSpace--Load: "+ns.name);
        }
    }
    @Override
    public void load(Document dou) throws Exception
    {
        this.load(dou, false);
    }
    public void load(String src, boolean slient) throws Exception
    {
        this.load(XML.getDocument(src), slient);
    }
    public void load(String src) throws Exception
    {
        this.load(XML.getDocument(src), false);
    }
    public void load(InputStream in, boolean slient) throws Exception
    {
        this.load(XML.getDocument(in), slient);
    }
    public void load(InputStream in) throws Exception
    {
        this.load(XML.getDocument(in), false);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="ResourceLink-Loader">
    public static final String DOCTYPE_RESOURCE_Link="ResourceLink";
    
    /**
     * <pre>
     * (1)The DocumentType of {@code Document dou} must be 'ResourceLink'
     * (2)The document list serveral paths of resources configuration 
     * (3)Then load the configuration files one another according to
     * the paths.
     * </pre>
     * @param <T>
     * @param dou
     * @param slient if slient==false: display a path listed in the ResourceLink
     * File,when the file is loaded successfully
     * @return 
     */
    @Passed
    public <T extends Factory> T loadAll(Document dou, boolean slient)
    {
        //check the type--------------------------------------------------------
        XML.checkDocType(dou, DOCTYPE_RESOURCE_Link);
        
        //load link-------------------------------------------------------------
        Node conf=dou.getElementsByTagName("configuration").item(0);
        if(conf==null) throw new TagMissedException("configuration");
        
        NodeList child=conf.getChildNodes();
        String name=null,url=null;
        Node cur=null;
        NamedNodeMap attr=null;
        for(int i=0,len=child.getLength();i<len;i++)
        {
            cur=child.item(i);
            if(cur.getNodeType()!=Node.ELEMENT_NODE) continue;
            name=cur.getNodeName();
            if(!name.equals("resource")) throw new UnknownTagException(name);
            attr=XML.getAttributes(cur);
            url=XML.getAttributeNoNull(attr, "url");
            try
            {
                this.load(url, slient);
                if(!slient) System.out.println("Succeed to load Resource: "+url);
            }
            catch(Exception e)
            {
                e.printStackTrace();
                System.err.println("Fail to load Resource: "+url);
            }
        }
        return (T) this;
    }
    public <T extends Factory> T loadAll(Document dou) throws Exception
    {
        return this.loadAll(dou, false);
    }
    public <T extends Factory> T loadAll(String src, boolean slient) throws Exception
    {
        return loadAll(XML.getDocument(src), slient);
    }
    public <T extends Factory> T loadAll(String src) throws Exception
    {
        return loadAll(XML.getDocument(src), false);
    }
    public <T extends Factory> T loadAll(InputStream in, boolean slient) throws Exception
    {
        return loadAll(XML.getDocument(in), slient);
    }
    public <T extends Factory> T loadAll(InputStream in) throws Exception
    {
        return loadAll(XML.getDocument(in), false);
    }
    //</editor-fold>
    //<editor-fold defaultstate="collapsed" desc="Instance">
    @Override
    public <T extends Initable> T instance(String namespace, String name) throws Exception
    {
        T r=null;
     
        Resource rs=this.getResource(namespace, name);
        if(rs==null) throw new NullPointerException();
        Constructor con=rs.getCons();
        r=(T) con.newInstance();
        r.init(rs.columns);
        
        return r;
    }
    public <T extends Initable> T instance(String namespace, String name, Consumer consumer) throws Exception
    {
        T r=null;
        
        Resource rs=this.getResource(namespace, name);
        if(rs==null) throw new NullPointerException();
        Constructor con=rs.getCons();
        r=(T) con.newInstance();
        consumer.accept(r);
        r.init(rs.columns);
        
        return r;
    }
    public <T extends Initable> T instance(String namespace, String name, Meta columns) throws Exception
    {
        T r=null;
        
        Resource rs=this.getResource(namespace, name);
        if(rs==null) throw new NullPointerException();
        Constructor con=rs.getCons();
        r=(T) con.newInstance();
        columns.putAllIfAbsent(rs.columns);
        r.init(columns);
        
        return r;
    }
     /**
      * <pre>
     * (1)Get resource with specific name from appointed Namespace, and
     * create an instance 
     * (2)add properties in {@code Meta columns} to default Meta of the
     * Resource, but will not replace the default properties
     * {@code columns.putAllIfAbsent(rs.columns);}
     * (3)Consumer.accept(instance)
     * (4)instance.init(columns)
     * </pre>
     * @param <T>
     * @param namespace
     * @param name
     * @param columns
     * @param consumer
     * @return
     * @throws Exception 
     */
    public <T extends Initable> T instance(String namespace, String name, Meta columns, Consumer consumer) throws Exception
    {
        T r=null;
       
        Resource rs=this.getResource(namespace, name);
        if(rs==null) throw new NullPointerException();
        Constructor con=rs.getCons();
        r=(T) con.newInstance();
        consumer.accept(r);
        columns.putAllIfAbsent(rs.columns);
        r.init(columns);
        
        return r;
    }
    /**
     * <pre>
     * [NameSpace].(the last '.')[Resource]
     * {@code 
     *      int split=code.lastIndexOf('.');
     *      namespace=code.substring(0,split);
     *      resource=code.substring(split+1);
     * }
     * </pre>
     * @param <T>
     * @param code
     * @return
     * @throws Exception 
     */
    public  <T extends Initable> T instance(String code) throws Exception 
    {
        int split=code.lastIndexOf('.');
        return this.instance(code.substring(0,split), code.substring(split+1));
    }
    public <T extends Initable> T instance(String code, Consumer con) throws Exception
    {
        int split=code.lastIndexOf('.');
        return this.instance(code.substring(0,split), code.substring(split+1), con);
    }
    public <T extends Initable> T instance(String code, Meta columns) throws Exception
    {
        int split=code.lastIndexOf('.');
        return this.instance(code.substring(0,split), code.substring(split+1), columns);
    }
    public <T extends Initable> T instance(String code, Meta columns, Consumer con) throws Exception
    {
        int split=code.lastIndexOf('.');
        return this.instance(code.substring(0,split), code.substring(split+1), columns, con);
    }
    //</editor-fold>
}
