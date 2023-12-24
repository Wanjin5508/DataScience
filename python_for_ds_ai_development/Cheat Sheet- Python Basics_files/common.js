const startAppRegex = /::startApplication(\ )?{([^{}]*)}/
const openFileRegex = /::openFile(\ )?{([^{}]*)}/
const openDatabaseRegex = /::openDatabase(\ )?{([^{}]*)}/
const openBigDataRegex = /::openBigDataTool(\ )?{([^{}]*)}/
const openCloudRegex = /::openCloudTool(\ )?{([^{}]*)}/
const openEmbeddableAIRegex = /::openEmbeddableAITool(\ )?{([^{}]*)}/
const pageRegex = /^::page(\ )?{([^{}]*)}$/

var currentPage;
let asset_library_prefix_url = "";

$.fn.nextUntilWithTextNodes = function (until) {
    var matched = $.map(this, function (elem, i, until) {
        var matched = [];
        while ((elem = elem.nextSibling) && elem.nodeType !== 9) {
            if (elem.nodeType === 1 || elem.nodeType === 3) {
                if (until && jQuery(elem).is(until)) {
                    break;
                }
                matched.push(elem);
            }
        }
        return matched;
    }, until);

    return this.pushStack(matched);
};

function capitalize(s) {
    return s && s[0].toUpperCase() + s.slice(1);
}

function setAssetLibraryPrefixUrl(url) {
    asset_library_prefix_url = url;
}

function getStructuredData(metadata) {
    let authors = metadata["author"]
    authors?.forEach(author => {
        // Default to Person if no type is specified
        author["@type"] = "Person" || author["@type"];
    });

    let structuredData = {
        "@context": "http://schema.org/",
        "@type": "Article"
    }

    metadata["headline"] && (structuredData["headline"] = metadata["headline"]);
    metadata["image"] && (structuredData["image"] = metadata["image"]);
    authors && (structuredData["author"] = authors);
    metadata["datePublished"] && (structuredData["datePublished"] = metadata["datePublished"]);

    return structuredData
}

function injectMetadata(metadata) {
    metadata = JSON.parse(metadata);
    let structuredData = getStructuredData(metadata);
    let head = $('head');
    let script = document.createElement('script');
    script.type = 'application/ld+json';
    script.text = JSON.stringify(structuredData);
    head.append(script);
}

function adjustCodeBlocks(doc) {
    doc.previewContainer.find("pre").each(function (i) {
        let pre = $(this);

        if (pre.context.nextSibling === null || pre.context.nextSibling.tagName !== 'BUTTON') {
            let lines = pre.find('ol.linenums');

            let content = '';
            let $lineNumbers = $("<ol class='formatted-line-numbers'></ol>");

            let lineIndex = 1
            lines.children('li').each(function (i) {
                // store code block content in a string
                content += (content === '') ? this.textContent : '\n' + this.textContent;

                // create a line number for each line of code
                $lineNumbers.append(`<li>${lineIndex}</li>`);
                lineIndex += 1
            });
            pre.prepend($lineNumbers);

            // if only one-line of code, add inline copy and execute buttons
            if (lines.children().length === 1) {
                pre.context.innerHTML += '<button title="Copy" class="action-code-block copy-code-block one-line"><i class="fa fa-copy" aria-hidden="true"></i><span class="popuptext" id=\"md-code-block-copy-' + i + '\">Copied!</span></button>';
                if (doc.settings.tool_type != "instructional-lab" && pre.find('code.language-bash, code.language-shell').length) pre.context.innerHTML += '<button title="Execute" class="action-code-block execute-code-block one-line"><i class="fa fa-terminal" aria-hidden="true"></i><span class="popuptext" id=\"md-code-block-execute-' + i + '\">Executed!</span></button>';
                pre.css("padding-right", "42px");
            } else {
                pre.context.innerHTML += '<button title="Copy" class="action-code-block copy-code-block multiple-lines"><i class="fa fa-copy" aria-hidden="true"></i><span class="popuptext" id=\"md-code-block-copy-' + i + '\">Copied!</span></button>';
                if (doc.settings.tool_type != "instructional-lab" && pre.find('code.language-bash, code.language-shell').length) pre.context.innerHTML += '<button title="Execute" class="action-code-block execute-code-block multiple-lines"><i class="fa fa-terminal" aria-hidden="true"></i><span class="popuptext" id=\"md-code-block-execute-' + i + '\">Executed!</span></button>';
            };

            // create and bind copy code button
            let copyButton = pre.find('button.action-code-block.copy-code-block');
            copyButton.bind(editormd.mouseOrTouch("click", "touchend"), function () {
                try {
                    navigator.clipboard.writeText(content);
                    let popup = $(`span.popuptext#md-code-block-copy-${i}`);
                    popup.toggleClass("show");
                    setTimeout(function () {
                        popup.toggleClass("show");
                    }, 1500);
                } catch (e) {
                    console.log(e);
                }
            });

            // create and bind execute code button
            let executeButton = pre.find('button.action-code-block.execute-code-block');
            executeButton.bind(editormd.mouseOrTouch("click", "touchend"), function () {
                try {
                    //sending request to the UI with type 'execute_code'
                    requestToUI({ type: "execute_code", command: content })
                    navigator.clipboard.writeText(content);
                    let popup = $(`span.popuptext#md-code-block-execute-${i}`);
                    popup.toggleClass("show");
                    setTimeout(function () {
                        popup.toggleClass("show");
                    }, 1500);
                } catch (e) {
                    console.log(e);
                }
            });
        }
    })
}

function findGetParameter(parameterName) {
    let result = null,
        tmp = [];
    let items = window.location.search.substr(1).split("&");
    for (var index = 0; index < items.length; index++) {
        tmp = items[index].split("=");
        if (tmp[0] === parameterName) result = decodeURIComponent(tmp[1]);
    }
    return result;
}

function parseDirective(context, required = []) {
    context = context.replace(/[\u2018-\u201B\u275B-\u275C]/g, "'")
    context = context.replace(/[\u201C-\u201F\u275D-\u275E]/g, '"')

    const regexFullMatch = /::\w+(\ )?{([^{}]*)}/;
    const regexKeyValue = /(\w+)="([^"]*)"/g;
    const m = context.match(regexFullMatch);

    if (m) {
        dic = Object.fromEntries(
            Array.from(m[2].matchAll(regexKeyValue), v => [v[1], v[2]])
        );
        const valid = required.every(val => Object.keys(dic).includes(val) && dic[val] != "")
        dic["valid"] = valid
        return dic
    } else {
        return { "valid": false }
    }
}

function isValidPageDirective(context) {
    if (pageRegex.test(context)) {
        const { valid } = parseDirective(context);
        return valid
    }
    return false
}

function fixTextAdjacentToDirective(doc) {
    let cm = doc.cm
    let txt = cm.getValue()
    let cursor = cm.getCursor();
    let changeOccured = false;
    let adjustCursor = false;

    //adding extra space below a directive at the beginning of the document when there is some adjacent text below the directive
    let pattern = /^(::\w+(\ )?{([^{}]*)}\n)([^\n])/g
    while (pattern.test(txt)) {
        txt = txt.replace(pattern, '$1\n$4')
        changeOccured = true;
        adjustCursor = true;
    }
    //adding extra space below a directive when there is some adjacent text below the directive
    pattern = /(\n::\w+(\ )?{([^{}]*)}\n)([^\n])/g
    while (pattern.test(txt)) {
        txt = txt.replace(pattern, '$1\n$4')
        changeOccured = true;
        adjustCursor = true;
    }
    //adding extra space above a directive when there is some adjacent text above the directive
    pattern = /([^\n])(\n::\w+(\ )?{([^{}]*)}\n)/g
    while (pattern.test(txt)) {
        txt = txt.replace(pattern, '$1\n$2')
        changeOccured = true;
    }

    //adding extra space below a directive when there is some adjacent text below the directive (In a list)
    pattern = /(\t*[^\n\t])(\n((\t)*)::\w+(\ )?{([^{}]*)}\n)/g
    while (pattern.test(txt)) {
        txt = txt.replace(pattern, '$1\n$3$2')
        changeOccured = true;
    }

    //adding extra space above a directive when there is some adjacent text above the directive (In a list)
    pattern = /(\n((\t)*)::\w+(\ )?{([^{}]*)}\n)(\t*([^\n\t]))/g
    while (pattern.test(txt)) {
        txt = txt.replace(pattern, '$1$2\n$6')
        changeOccured = true;
    }

    if (changeOccured) {
        cm.setValue(txt)
        cm.refresh();
        cm.setCursor(adjustCursor ? cursor.line + 1 : cursor.line, cursor.ch)
    }
    return changeOccured
}

function handleDynamicHeader(doc) {
    let currentlyParsing = false;
    let foundHeader = false;
    doc.previewContainer.find("p").each(function () {
        let tag = $(this);
        let content = tag.context.innerHTML;
        if (content.startsWith("::header") && content.includes("start")) {
            const { fixed, valid } = parseDirective(content, ["fixed"]);
            if (valid && fixed === "false") {
                currentlyParsing = true;
            }
        } else if (content.startsWith("::header") && content.includes("end")) {
            const { valid } = parseDirective(content, []);
            if (valid && currentlyParsing === true) {
                currentlyParsing = false;
                foundHeader = true;
            }
        }
    })
    return foundHeader;
}

function handleStaticHeader(doc) {
    let headerContent = "";
    let stickyHeader = doc.preview.find(".sticky-header");
    stickyHeader.empty();
    let foundHeader = false;
    let markdown = document.getElementsByClassName('editormd-markdown-textarea')[0].innerText;
    let markdownArray = markdown.split("\n");
    let content = "";
    let currentlyParsing = false;
    let codeblocks = markdownArray.forEach(function (line) {
        if (foundHeader) {
            return true;
        }
        if (line && line.length > 0 && line.startsWith("::header") && line.includes("start")) {
            const { fixed, valid } = parseDirective(line, ["fixed"]);
            if (valid && fixed === "true") {
                currentlyParsing = true;
            }

        } else if (line && line.length > 0 && line.startsWith("::header") && line.includes("end")) {
            const { valid } = parseDirective(line, []);
            if (valid && currentlyParsing === true) {
                currentlyParsing = false;
                if (content) {
                    console.log("updating content", content)
                    // if content contain "!" and "(" and "[" and ")" and "]", replace everything in between "(" and ")" with "test"
                    if (content.includes("!") && content.includes("(") && content.includes(")") && content.includes("[") && content.includes("]")) {
                        //get eveything that is between "(" and ")"
                        let contentBetweenParenthesis = content.substring(content.indexOf("(") + 1, content.indexOf(")"));
                        // create a new url by appending contentBetweenParenthesis to the asset_library_prefix_url
                        let newSource = "(" + asset_library_prefix_url + "/" + contentBetweenParenthesis + ")";
                        content = content.replace(/\(.*\)/g, newSource);
                    }
                    stickyHeader.append(editormd.$marked.parse(content));
                    stickyHeader.css("padding-top", "20px");
                    stickyHeader.css("padding-bottom", "20px");
                    foundHeader = true;
                    return true;
                }

            }


        } else if (line && line.length > 0 && currentlyParsing === true) {
            content += line + "\n";
        }
    })
    return foundHeader
}

function cleanStaticHeader(doc) {
    doc.previewContainer.contents().each(function () {
        let tag = $(this);
        let content = tag.context.innerHTML;
        if (pageRegex.test(content)) {
            return false;
        } else {
            tag.remove();
            tag.context.innerHTML = "";
        }
    })
}

function cleanDynamicHeader(doc) {
    let currentlyParsing = false;
    let firstHeaderFound = false;
    let pageFound = false;
    doc.previewContainer.contents().each(function () {
        let tag = $(this);
        let content = tag.context.nodeType == 3 ? "" : tag.context.innerHTML;
        if (pageFound) {
            return false;
        }
        if (pageRegex.test(content)) {
            pageFound = true;
            return false;
        } else if (firstHeaderFound && currentlyParsing === false) {
            tag.remove();
            tag.context.innerHTML = "";
        } else if (content.startsWith("::header") && content.includes("start") && content.includes("false") && firstHeaderFound === false) {
            const { fixed, valid } = parseDirective(content, ["fixed"]);
            if (valid && fixed === "false") {
                currentlyParsing = true;
                tag.remove();
            }
        } else if (content.startsWith("::header") && content.includes("end")) {
            const { valid } = parseDirective(content, []);
            if (valid && currentlyParsing === true) {
                currentlyParsing = false;
                firstHeaderFound = true;
            }
            tag.remove();
        } else if (currentlyParsing === false) {
            tag.remove()
            tag.context.innerHTML = "";
        }
    })

}

function fixEncoding(doc) {

    var Utf8 = {
        decode: function (utftext) {
            return JSON.parse( JSON.stringify( utftext ) );
        }
    }

    doc.previewContainer.contents().each(function () {
        let tag = $(this);

        let content = tag.context.innerHTML;

        if (content !== undefined) {

            decodedContent = Utf8.decode(content);

            if (content !== decodedContent) {
                tag.context.innerHTML = Utf8.decode(content) + "<i class=\"fa fa-times-circle-o\" style=\"color: red; font-size: 2em; margin-left: 1.25rem;\" title=\"This text contains invalid characters and may not save as expected\"></i>";
            }
        }
    })

}

function fixCustomPlugins(doc) {
    //setup headers
    let stickyHeader = doc.preview.find(".sticky-header");
    stickyHeader.empty();
    stickyHeader.css("padding-top", "0px");
    stickyHeader.css("padding-bottom", "0px");
    stickyHeader.css("height", "auto");
    dynamicHeaderFound = handleDynamicHeader(doc);
    if (dynamicHeaderFound) {
        //only let the first dynamic header
        cleanDynamicHeader(doc);

    } else {
        //only let the first static header
        handleStaticHeader(doc);
        cleanStaticHeader(doc);
    }
    // setup for pagination custom plugin
    let pages = 0; // running counter of pages
    let header = doc.preview.find('.md-header');
    let pagination = header.find('.pagination');
    let tableOfContents = doc.preview.find('.table-of-contents');
    let pageDirectives = $("p").filter((i, elem) => isValidPageDirective(elem.innerHTML)) //all valid page directives
    $("button.toc").css("visibility", pageDirectives.length <= 0 ? "hidden" : "");

    currentPage = pageDirectives.length <= 0 ? -1 : 0

    tableOfContents.get(0).innerHTML = ""
    pagination.get(0).innerHTML = ""

    if (pageDirectives.length <= 0) {
        doc.previewContainer.append(`<h1>Markdown-based Instructions should include pages</h1><p>Please make sure to include at least one <strong>page</strong> directive in your instructions. Use the <strong>New Page</strong> button from the toolbar to create pages.</p><h3 align="center"> © IBM Corporation ${new Date().getFullYear()}<h3/>`)
    }

    //main renderer logic
    doc.previewContainer.contents().each(function () {
        let tag = $(this);
        let content = tag.context.innerHTML;
        if (startAppRegex.test(content)) {
            const { port, display, name, route, valid } = parseDirective(content, ["port", "display", "name", "route"]);
            if (valid) {
                tag.context.innerHTML =
                    `<button class="plugin" onclick="launchApplication('${port}','${route}','${display}')"><i class = "fa fa-rocket"></i> <span class = "plugin-text">${name}</span></button>`
            }
            else {
                tag.remove()
            }
        } else if (openFileRegex.test(content)) {
            const { path, valid } = parseDirective(content, ["path"]);
            if (valid) {
                const file = path.substring(path.lastIndexOf("/") + 1)
                tag.context.innerHTML =
                    `<button class="plugin" onclick="openFile('${path}')"><i class="fa fa-file-text"></i> <span class="plugin-text">Open <strong>${file}</strong> in IDE</span></button>`
            }
            else {
                tag.remove()
            }
        } else if (openDatabaseRegex.test(content)) {
            const { db, start, valid } = parseDirective(content, ["db", "start"]);
            if (valid) {
                tag.context.innerHTML =
                    `<button class="plugin" onclick="openDataBase('${db}','${start}')"><i class="fa fa-database"></i> <span class="plugin-text">${start === 'true' ? "Open and Start " + db + " in IDE" : "Open " + db + " Page in IDE"}</span></button>`
            }
            else {
                tag.remove()
            }
        } else if (pageRegex.test(content)) {
            const { title, valid } = parseDirective(content);
            if (valid) {
                paginate(tag, title, pages, pageDirectives, pagination, tableOfContents);
                pages++;
            }
            tag.remove()
        } else if (openBigDataRegex.test(content)) {
            const { tool, start, valid } = parseDirective(content, ["tool", "start"]);
            if (valid) {
                tag.context.innerHTML =
                    `<button class="plugin" onclick="openBigData('${tool}','${start}')"><i class="fa fa-tasks"></i> <span class="plugin-text">${start === 'true' ? "Open and Start " + tool + " in IDE" : "Open " + tool + " in IDE"}</span></button>`
            }
            else {
                tag.remove()
            }
        } else if (openCloudRegex.test(content)) {
            const { tool, action, label, valid } = parseDirective(content, ["tool", "action", "label"]);
            if (valid) {
                const actionText = (action === "none") ? `Open ${tool}` : label;
                const text = `${actionText} in IDE`;
                tag.context.innerHTML =
                    `<button class="plugin" onclick="openCloud('${tool}','${action}')"><i class="fa fa-tasks"></i> <span class="plugin-text">${text}</span></button>`
            }
            else {
                tag.remove()
            }
        } else if (openEmbeddableAIRegex.test(content)) {
            const { tool, action, label, valid } = parseDirective(content, ["tool", "action", "label"]);
            if (valid) {
                const actionText = (action === "none") ? `Open ${tool}` : label;
                const text = `${actionText} in IDE`;
                tag.context.innerHTML =
                    `<button class="plugin" onclick="openEmbeddableAI('${tool}','${action}')"><i class="fa fa-tasks"></i> <span class="plugin-text">${text}</span></button>`
            }
            else {
                tag.remove()
            }
        }
    })
    // bind buttons for paginnation after the renderer is done
    fixPaginationBindings(doc)
}

function headerContent() {
    let markdown = document.getElementsByClassName('editormd-markdown-textarea')[0].innerText;
    let foundHeader = false;
    let markdownArray = markdown.split("\n");
    let content = "";
    let currentlyParsing = false;
    let codeblocks = markdownArray.forEach(function (line) {
        if (foundHeader) {
            return content;
        }
        if (line && line.length > 0 && line.startsWith("::header") && line.includes("start")) {
            const { fixed, valid } = parseDirective(line, ["fixed"]);
            if (valid) {
                currentlyParsing = true;
            }

        } else if (line && line.length > 0 && line.startsWith("::header") && line.includes("end")) {
            const { valid } = parseDirective(line, []);
            if (valid && currentlyParsing === true) {
                currentlyParsing = false;
                foundHeader = true;
                return content;
            }


        } else if (line && line.length > 0 && currentlyParsing === true) {
            content += line + "\n";
        }
    })
    return content;

}

function removeHeaders(cm) {
    let txt = cm.getValue()
    // find the index of ::page in txt
    let index = txt.indexOf("::page");
    // replace the string with new string from index to end
    if (index > -1) {
        txt = txt.substring(index);
    }
    // save the doc and replace it with txt
    cm.setValue(txt);
    cm.save();
}

function fixPaginationBindings(doc) {
    const tableOfContentsMenu = $('.table-of-contents');

    // bind pagination to buttons and links
    $('.pagination').children(".page-item").bind(editormd.mouseOrTouch("click", "touchend"), function () {
        changePage($(this).index() / 2, doc); // exclude page dividers from the index count
        doc.previewContainer.scrollTop(0);
    });
    $('.table-of-contents').children("button").bind(editormd.mouseOrTouch("click", "touchend"), function () {
        changePage($(this).index(), doc);
        tableOfContentsMenu.toggleClass('opened');
        doc.previewContainer.scrollTop(0);
    });
    $('div.page-footer > button.next').bind(editormd.mouseOrTouch("click", "touchend"), function () {
        let currPage = $(this).parents('.page').index();
        changePage(currPage + 1, doc);
        doc.previewContainer.scrollTop(0);
    });
    $('div.page-footer > button.previous').bind(editormd.mouseOrTouch("click", "touchend"), function () {
        let currPage = $(this).parents('.page').index();
        changePage(currPage - 1, doc);
        doc.previewContainer.scrollTop(0);
    });

    // wrap all pages in a parent container
    $(".page").wrapAll('<div class="pages"></div>');
    doc.previewContainer.css("height", "100%");

    // set progress bar
    setProgressBarWidth(currentPage);
    $('.page-divider').css("background-color", doc.previewContainer.css('background-color'));
}

function paginate(tag, title, pages, pageDirectives, pagination, tableOfContents) {
    if (title) {
        tag.after('<h1 class="pageTitle">' + title + '</h1>')
    }
    let allElements = tag.nextUntilWithTextNodes(pageDirectives);

    // create a page link in the progress bar
    if (pages <= currentPage) {
        pagination.append('<li class="page-item active"></li>');
    } else {
        pagination.append('<li class="page-item"></li>');
    };
    pagination.append('<li class="page-divider"></li>');


    // create a corresponding page and an entry in table of contents
    if (pages === currentPage) {
        if (allElements.length != 0) allElements.wrapAll("<div class='page' id=\"page-" + (pages + 1) + "\" />");
        else tag.after("<div class='page' id=\"page-" + (pages + 1) + "\" /></div>")
        tableOfContents.append('<button class="chapter active"><span class="chapter-num">' +
            (pages + 1) + '</span><span class="chapter-title">' + (title ? title : "") + '</span></button >');
    } else {
        if (allElements.length != 0) allElements.wrapAll("<div class='page' id=\"page-" + (pages + 1) + "\" style='display:none;' />");
        else tag.after("<div class='page' id=\"page-" + (pages + 1) + "\" style='display:none;' /></div>")
        tableOfContents.append('<button class="chapter"><span class="chapter-num">' +
            (pages + 1) + '</span><span class="chapter-title">' + (title ? title : "") + '</span></button >');
    };


    // add a previous and next buttom at bottom of each page
    let page = $('#page-' + (pages + 1));
    page.append('<div class="page-footer"></div>');
    let footer = page.find('.page-footer');
    if (pages === 0) {
        if (pageDirectives.length > 1) {
            footer.append('<button class="hidden"></button>');
            footer.append('<button class="next">Next</button>');
        }
    } else if (pages === pageDirectives.length - 1) {
        footer.append('<button class="previous">Previous</button>');
    } else {
        footer.append('<button class="previous">Previous</button>');
        footer.append('<button class="next">Next</button>');
    };
}

function adjustProgressBarWidth() {
    let currentPage = $("div.table-of-contents > button.chapter.active > span.chapter-num").text()
    if (currentPage) {
        setProgressBarWidth(parseInt(currentPage) - 1)
    }
}

function setPreviewWatchToolbar(doc) {

    // create header container
    doc.preview.prepend('<div class="sticky-header" style="z-index:1;"></div>');
    doc.preview.prepend('<div class="md-header" style="z-index:1;"></div>');
    let header = doc.preview.find('.md-header');

    // create the toolbar and pagination containers
    header.prepend(
        '<div class="toolbar"></div> \
        <nav class="instruction-progress-bar" aria-label="pagination"> \
            <div class="instruction-progress-fill"></div> \
            <ul class="pagination"></ul> \
        </nav>'
    );
    let toolbar = header.find('.toolbar');

    // populate toolbar
    toolbar.append('<button class="toc"><i class="fa fa-bars" aria-hidden="true" style="margin-right: 15px; display: inline;"></i><p>Table of Contents</p></button>');
    let toc = toolbar.find('.toc')
    toc.css("visibility", "hidden");
    toc.bind(editormd.mouseOrTouch("click", "touchend"), function () {
        const tableOfContentsMenu = $('.table-of-contents');
        tableOfContentsMenu.toggleClass('opened');

        // dismiss table of contents menu if you click on anything outside of it
        const outsideClickListener = (event) => {
            const target = $(event.target);
            if (!target.closest(this).length && !target.closest(tableOfContentsMenu).length && tableOfContentsMenu.hasClass('opened')) {
                tableOfContentsMenu.toggleClass('opened');
                removeClickListener();
            };
        };
        const removeClickListener = () => {
            document.removeEventListener('click', outsideClickListener);
        }

        if (tableOfContentsMenu.hasClass('opened')) {
            document.addEventListener('click', outsideClickListener);
        } else {
            removeClickListener();
        }
    });

    toolbar.append(`<div class="tools">
        ${doc.settings.tool_type == "instructional-lab" ? '<button value="on" class="tool-icon" id="togglePreviewTheme" title="Toggle preview theme "><i class="fa fa-sun-o mr-1" aria-hidden="true"></i><i class="fa fa-toggle-on" aria-hidden="true"></i></button>' : ""}
        <button class="tool-icon" id="print" title="Print instructions"><i class="fa fa-print" aria-hidden="true"></i></button>
        <button class="tool-icon" id="font-up" title="Increase font size"><i class="fa fa-font" aria-hidden="true"></i><sup>+</sup></button>
        <button class="tool-icon" id="font-down" title="Decrease font size"><i class="fa fa-font" aria-hidden="true"></i><sup>-</sup></button>
    </div>`);

    let toolIcons = toolbar.find('.tools');

    if (toolIcons) {
        toolIcons.find("#print").bind(editormd.mouseOrTouch("click", "touchend"), function () {
            let instructions = '';
            doc.previewContainer.children().each(function () {
                if (!$(this).hasClass("pages")) instructions += $(this)[0].outerHTML;
                else {
                    $(this).find(".page").each(function () {
                        $(this).children().each(function () {
                            if (!$(this).hasClass("page-footer")) instructions += $(this)[0].outerHTML;
                        })
                    })
                }
            });
            const content = window.open('', '', 'height=500, width=500');
            content.document.write('<html>');
            content.document.write('<body >');
            content.document.write(instructions);
            content.document.write('</body></html>');
            content.document.close();
            content.print();
        });
        toolIcons.find("#font-up").bind(editormd.mouseOrTouch("click", "touchend"), function () {
            doc.preview.find("div:not(.toolbar) > *").css('font-size', '+=1');
        });
        toolIcons.find("#font-down").bind(editormd.mouseOrTouch("click", "touchend"), function () {
            doc.preview.find("div:not(.toolbar) > *").css('font-size', '-=1');
        });

        if (doc.settings.tool_type == "instructional-lab") {
            toolIcons.find("#togglePreviewTheme").bind(editormd.mouseOrTouch("click", "touchend"), function () {
                if (this.value === "off") {
                    this.value = "on";
                    this.innerHTML = '<i class="fa fa-sun-o mr-1" aria-hidden="true"></i><i class="fa fa-toggle-on" aria-hidden="true"></i>';
                    doc.setPreviewTheme('default');
                } else {
                    this.value = "off";
                    this.innerHTML = '<i class="fa fa-moon-o mr-1" aria-hidden="true"></i><i class="fa fa-toggle-off" aria-hidden="true"></i>';
                    doc.setPreviewTheme('dark');
                }
            });
        }
    };

    // create table of contents
    $(header).after('<div class="table-of-contents"></div>');
}

function setProgressBarWidth(pageNumber) {
    if (pageNumber == -1) {
        $('.instruction-progress-fill').css("width", "0px");
    }
    else {
        const rectInfo = $('.page-item')[0].getBoundingClientRect();
        let width;
        if (rectInfo.width) {
            // `width` is available for IE9+
            width = rectInfo.width;
        } else {
            // Calculate width for IE8 and below
            width = rectInfo.right - rectInfo.left;
        }
        let progress = ((pageNumber + 1) * width) + (pageNumber * 2);
        $('.instruction-progress-fill').width(progress.toString() + "px");

    }
}

function changePage(chosenPageNumber, doc, scroll = true) {
    // set active class indicators
    $('.pagination').find("li.active").removeClass('active');
    const activePage = $('.pagination').children(".page-item").eq(chosenPageNumber);
    activePage.addClass('active');
    $('.table-of-contents').find("button.active").removeClass('active');
    $('.table-of-contents').children("button").eq(chosenPageNumber).addClass('active');

    setProgressBarWidth(chosenPageNumber);

    // show the active page
    doc.previewContainer.find(`div#page-${currentPage + 1}`).hide();
    doc.previewContainer.find(`div#page-${chosenPageNumber + 1}`).show();

    // store current page number
    currentPage = chosenPageNumber;

    // scroll to corresponding editor line
    if (scroll && !doc.settings.readOnly) {
        let positionOfCurrentPageDirectiveInEditor = doc.cm.charCoords({ line: doc.settings.pageMap[currentPage], ch: 0 }, "local").top + 1;
        doc.cm.scrollTo(null, positionOfCurrentPageDirectiveInEditor)
    }
}

function setVisible(selector, visible) {
    document.querySelector(selector).style.display = visible ? 'block' : 'none';
}

