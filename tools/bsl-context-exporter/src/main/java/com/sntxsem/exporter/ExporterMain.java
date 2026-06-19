package com.sntxsem.exporter;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.github._1c_syntax.bsl.context.api.Context;
import com.github._1c_syntax.bsl.context.api.ContextMethod;
import com.github._1c_syntax.bsl.context.api.ContextProperty;
import com.github._1c_syntax.bsl.context.api.ContextType;
import com.github._1c_syntax.bsl.context.platform.PlatformContextGrabber;
import com.github._1c_syntax.bsl.context.platform.ShlangParser;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.io.BufferedWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;

@Command(name = "bsl-context-exporter", mixinStandardHelpOptions = true)
public class ExporterMain implements Callable<Integer> {

    @Option(names = "--hbk-dir", required = true, description = "Directory with shcntx/shlang HBK files")
    Path hbkDir;

    @Option(names = "--output", required = true, description = "Output directory")
    Path output;

    @Option(names = "--platform-version", defaultValue = "8.3.27")
    String platformVersion;

    private final ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

    public static void main(String[] args) {
        int code = new CommandLine(new ExporterMain()).execute(args);
        System.exit(code);
    }

    @Override
    public Integer call() throws Exception {
        Files.createDirectories(output);
        List<Map<String, Object>> chunks = new ArrayList<>();

        Path shcntxRu = hbkDir.resolve("shcntx_ru.hbk");
        Path shcntxRoot = hbkDir.resolve("shcntx_root.hbk");
        if (Files.isRegularFile(shcntxRu)) {
            var grabber = PlatformContextGrabber.fromHbk(shcntxRu, shcntxRoot);
            var provider = grabber.parseBilingual();
            for (Context ctx : provider.getContexts()) {
                chunks.add(contextToChunk(ctx, "platform_api", "shcntx_ru.hbk", "ru"));
            }
        }

        Path shlangRu = hbkDir.resolve("shlang_ru.hbk");
        Path shlangRoot = hbkDir.resolve("shlang_root.hbk");
        if (Files.isRegularFile(shlangRu)) {
            var shlang = new ShlangParser().parse(shlangRu, shlangRoot);
            for (var entry : shlang.getKeywords().entrySet()) {
                var kw = entry.getValue();
                Map<String, Object> chunk = baseChunk(
                        "bsl_lang:shlang:" + entry.getKey(),
                        "bsl_lang",
                        "shlang_ru.hbk",
                        "ru"
                );
                chunk.put("title_ru", kw.getNameRu());
                chunk.put("title_en", kw.getNameEn());
                chunk.put("entity_kind", kw.getKind().name().toLowerCase());
                chunk.put("text", kw.getDescriptionRu() != null ? kw.getDescriptionRu() : "");
                chunk.put("syntax", kw.getSnippetRu() != null ? kw.getSnippetRu() : "");
                chunks.add(chunk);
            }
        }

        Path jsonl = output.resolve("bsl_chunks.jsonl");
        try (BufferedWriter writer = Files.newBufferedWriter(jsonl)) {
            for (Map<String, Object> chunk : chunks) {
                writer.write(mapper.writeValueAsString(chunk));
                writer.newLine();
            }
        }
        System.out.println("Exported " + chunks.size() + " chunks to " + jsonl);
        return 0;
    }

    private Map<String, Object> contextToChunk(Context ctx, String domain, String source, String locale) {
        String id = domain + ":" + ctx.getName();
        Map<String, Object> chunk = baseChunk(id, domain, source, locale);
        chunk.put("title_ru", ctx.getName());
        chunk.put("title_en", ctx.getNameEn() != null ? ctx.getNameEn() : ctx.getName());
        chunk.put("entity_kind", ctx.getKind().name().toLowerCase());

        StringBuilder text = new StringBuilder();
        if (ctx instanceof ContextType type) {
            if (type.getDescriptionRu() != null) {
                text.append(type.getDescriptionRu());
            }
            for (ContextMethod method : type.getMethods()) {
                text.append("\n\n## ").append(method.getName()).append("\n");
                if (method.getDescriptionRu() != null) {
                    text.append(method.getDescriptionRu());
                }
            }
            for (ContextProperty property : type.getProperties()) {
                text.append("\n\n## ").append(property.getName()).append("\n");
                if (property.getDescriptionRu() != null) {
                    text.append(property.getDescriptionRu());
                }
            }
        } else if (ctx instanceof ContextMethod method) {
            if (method.getDescriptionRu() != null) {
                text.append(method.getDescriptionRu());
            }
        }
        chunk.put("text", text.toString().trim());
        return chunk;
    }

    private Map<String, Object> baseChunk(String id, String domain, String source, String locale) {
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("domain", domain);
        chunk.put("title_ru", "");
        chunk.put("title_en", "");
        chunk.put("path", "");
        chunk.put("parent_path", "");
        chunk.put("entity_kind", "type");
        chunk.put("platform_version", platformVersion);
        chunk.put("hbk_source", source);
        chunk.put("locale", locale);
        chunk.put("text", "");
        chunk.put("syntax", "");
        chunk.put("parameters", "");
        chunk.put("signature", "");
        chunk.put("html_path", "");
        return chunk;
    }
}
