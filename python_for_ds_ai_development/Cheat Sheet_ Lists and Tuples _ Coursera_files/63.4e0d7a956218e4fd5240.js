(window.webpackJsonp=window.webpackJsonp||[]).push([[63],{"+tS/":function(module,exports,e){},"3//P":function(module,e,t){"use strict";t.r(e);var n,s,i,o=t("VkAN"),r=t.n(o),a=t("lSNA"),c=t.n(a),u=t("VbXa"),l=t.n(u),d=t("AeFk"),p=t("q1tI"),m=t("17x9"),b=t.n(m),f=t("F/us"),O=t.n(f),g=t("+LJP"),h=t("6/Gu"),j=t("2sch"),y=t("SJ7i"),v=t("BVC1"),x=t("kgYC"),C=t("eJMc"),k=t.n(C),A=t("wd/R"),S=t.n(A),N=t("kvW3"),T=t("8Hdl"),w=t("RT5p"),L=t("loer"),D=t("Fmrb"),I=t.n(D);var forumList_ForumsLabel=function(e){var t=e.discussionsLink,o=e.title,a=e.description,c=e.lastAnsweredAt,u=e.forumQuestionCount,l=Object(x.a)(),p="number"==typeof u,m=!!c,b=1===u?I()("thread"):I()("threads"),f=a.definition.value,O=f.startsWith("<co-content>")&&f.endsWith("</co-content>"),g=!w.c.isEmpty(a)&&O,j="Forum: ".concat(o,"\n                       ").concat(g?w.c.getInnerText(a):"","\n                       ").concat(m?"Last Post: ".concat(S()(c).fromNow()):"","\n                       ").concat(p?"".concat(u," ").concat(b):"");return Object(d.d)(k.a,{className:"rc-ForumsLabel nostyle",to:t,"aria-label":j},Object(d.d)("div",null,Object(d.d)(h.a,{container:!0,xs:12},Object(d.d)(h.a,{item:!0,xs:7,css:Object(d.c)(n||(n=r()(["\n              margin-bottom: ",";\n            "])),l.spacing(16))},Object(d.d)(T.a,{variant:"h2semibold",component:"p",css:Object(d.c)(s||(s=r()(["\n                padding: 0;\n              "])))},o),!w.c.isEmpty(a)&&Object(d.d)(L.a,{cml:a,isCdsEnabled:!0}),m&&Object(d.d)(T.a,{variant:"body2",color:"supportText",component:"span"},Object(d.d)(N.b,{message:I()("Last post {timeAgo}"),timeAgo:S()(c).fromNow()}))),p&&Object(d.d)(h.a,{container:!0,item:!0,xs:5,spacing:8,css:Object(d.c)(i||(i=r()(["\n                justify-content: flex-end;\n                align-items: center;\n\n                @media (max-width: 320px) {\n                  margin: 24px 0 0 -4px;\n                }\n              "])))},Object(d.d)(h.a,{item:!0},Object(d.d)(T.a,{className:"threads-count"},"".concat(u," ").concat(b)))))))};function ContentBlock(e){var t=e.children,n=e.className,s=Object(x.a)(),i={border:"1px solid ".concat(s.palette.gray[300])};return Object(d.d)("section",{className:n,css:i},t)}var F,q,forumList_ForumsList=function(e){return p.createElement(h.a,{container:!0,xs:12,"data-e2e":e.className},p.createElement(h.a,{item:!0,xs:12,component:ContentBlock},p.createElement("div",null,e.children)))},P=t("Jrms");t("a9ka");var R,V=Object(f.compose)(Object(P.k)({fields:["link","title","description","lastAnsweredAt","forumQuestionCount","parentForumId"]}))((function(e){var t=e.courseForums,n=e.mentorForums,s=e.courseForumStatistics,i=e.courseSlug,o=t.find((function(e){return e.forumType.typeName===P.p.rootForumType})),a=Object(x.a)(),c=n.filter((function(e){return!e.parentForumId})),u=t.filter((function(e){return e.parentForumId===o.id&&e.forumType.typeName===P.p.customForumType})),l=c.concat(u).map((function(e){var t=s.find((function(t){return t.courseForumId===e.id}));return Object.assign(e,{discussionsLink:v.a.join(Object(P.n)(i),e.link),lastAnsweredAt:t&&t.lastAnsweredAt,forumQuestionCount:t&&t.forumQuestionCount})}));return 0===l.length?null:Object(d.d)(forumList_ForumsList,{className:"rc-DiscussionsCourseForums"},Object(d.d)("ul",{css:Object(d.c)(F||(F=r()(["\n          list-style: none;\n          margin: 0;\n          padding: 0;\n        "])))},l.map((function(e){return Object(d.d)("li",{key:e.id,css:Object(d.c)(q||(q=r()(["\n              border-bottom: 1px solid ",";\n              padding: ",";\n\n              &:hover {\n                background-color: ",";\n              }\n            "])),a.palette.gray[300],a.spacing(24),a.palette.blue[50])},Object(d.d)(forumList_ForumsLabel,{discussionsLink:e.discussionsLink,title:e.title,description:e.description,lastAnsweredAt:e.lastAnsweredAt,forumQuestionCount:e.forumQuestionCount}))}))))}));t("JBNF");var B,E,personalization_DiscussionsLandingPageHeader=function(){var e=Object(x.a)();return Object(d.d)("div",{className:"rc-DiscussionsLandingPageHeader "},Object(d.d)("div",{css:Object(d.c)(R||(R=r()(["\n          padding: ",";\n        "])),e.spacing(32,0,32,0))},Object(d.d)(T.a,{variant:"h1semibold"},I()("Forums"))))},W=t("TSYQ"),Y=t.n(W),M=t("Ys1X"),H=(t("HkWL"),function(e){function DiscussionsGroupForums(){return e.apply(this,arguments)||this}return l()(DiscussionsGroupForums,e),DiscussionsGroupForums.prototype.render=function(){var e=this.props,t=e.groupForums,n=e.groupForumStatistics,s=e.courseSlug;if(!t||!t.length)return null;var i=t.map((function(e){var t=n.find((function(t){return t.id===e.id}));return{id:e.id,discussionsLink:v.a.join(Object(M.c)(s),e.link),title:e.title,description:e.description,lastAnsweredAt:t&&t.lastAnsweredAt,forumQuestionCount:t&&t.forumQuestionCount}}));return Object(d.d)(forumList_ForumsList,{className:"rc-DiscussionsGroupForums"},Object(d.d)(T.a,{component:"div",css:Object(d.c)(B||(B=r()(["\n            padding: 24px;\n          "])))},Object(d.d)("ul",{css:Object(d.c)(E||(E=r()(["\n              padding: 0;\n              margin: 0;\n              list-style-type: none;\n            "])))},i.map((function(e){return Object(d.d)("li",{key:e.id},Object(d.d)(forumList_ForumsLabel,{discussionsLink:e.discussionsLink,title:e.title,description:e.description,lastAnsweredAt:e.lastAnsweredAt,forumQuestionCount:e.forumQuestionCount}))})))))},DiscussionsGroupForums}(p.Component)),U=O.a.compose(Object(P.k)({fields:["link","title","description","parentForumId"]}))(H),Q=t("J4zp"),J=t.n(Q),K=t("3tO9"),G=t.n(K),X=t("MnCE"),z=t("Gok7"),Z=t("gNwb"),ee=t("TmOg"),te={ALL:"all",FORYOU:"foryou",ACTIVITY:"activity"},ne=Object(Z.e)({type:"BUTTON"})(k.a),DiscussionsLandingPageTabs_tabsStyles=function(e){return{".discussionsLandingPageTabs":{borderBottom:"1px solid ".concat(e.palette.gray[300]),justifyContent:"space-between"},"ul.tabs li.colored-tab.selected":{borderBottom:"4px solid ".concat(e.palette.blue[600]),padding:"".concat(e.spacing(0,0,8,0))},"ul.tabs li.colored-tab a.selected":G()({border:"none",color:e.palette.blue[600]},e.typography.h3bold),"ul.tabs li.colored-tab a:hover":{borderLeft:"none"},"ul.tabs li.colored-tab a":G()({padding:"".concat(e.spacing(0,4,0,0)),borderBottom:"4px solid transparent",borderLeft:"none"},e.typography.body1)}};function getTabs(e,t){var n,s=(n={},c()(n,t.FORYOU,{title:I()("Posts for you"),pathname:Object(P.l)(t.FORYOU),query:"",isActive:!1,tabKey:t.FORYOU}),c()(n,t.ALL,{title:I()("All forums"),pathname:Object(P.l)(t.ALL),query:"",isActive:!1,tabKey:t.ALL}),c()(n,t.ACTIVITY,{title:I()("Your activity"),pathname:Object(P.l)(t.ACTIVITY),query:"",isActive:!1,tabKey:t.ACTIVITY}),n),i=[te.ALL,te.FORYOU,te.ACTIVITY],o=function(e){var t=e.router,n=e.tabs,s=null;return n.forEach((function(e){t.isActive(Object(P.l)(e),!0)&&(s=e)})),s}({router:e,tabs:i})||i[0];return{activeTab:o,orderedTabs:i.map((function(e){var t=Object.assign({},s[e]);return e===o&&(t.isActive=!0),t}))}}var onTabEnter=function(e,t){e(t["data-link"])},DiscussionsLandingPageTabs_Tab=function(e){var t=e.title,n=e.pathname,s=e.query,i=e.isActive,o=e.tabKey,r={query:s,pathname:n};return Object(d.d)(p.Fragment,null,i&&Object(d.d)(z.a,{role:"status","aria-live":"assertive"},Object(d.d)("div",null," ",I()("Selected"))),Object(d.d)("li",{className:Y()("colored-tab",{selected:i}),key:t,role:"none",id:"tab-".concat(o)},Object(d.d)(ne,{to:r,className:i?"selected":void 0,trackingName:"track_active_forum_tabs",role:"tab","aria-selected":i},t)))},DiscussionsLandingPageTabs_TabPanel=function(e){var t=e.activeTab,n=e.children,s=e.tabName,i=Object(p.useState)(-1),o=J()(i,2),r=o[0],a=o[1];return Object(p.useEffect)((function(){t===s&&a(0)}),[]),t===s?Object(d.d)("div",{id:"panel-".concat(s),role:"tabpanel",tabIndex:r},n):null};var se=Object(ee.a)((function(e){var t=e.children,n=Object(x.a)();return Object(d.d)(h.a,{container:!0,item:!0,xs:12,css:DiscussionsLandingPageTabs_tabsStyles(n)},Object(d.d)(h.a,{container:!0,className:"discussionsLandingPageTabs"},Object(d.d)(h.a,{item:!0},Object(d.d)("ul",{role:"tablist",className:"tabs horizontal-box"},t))))}));var ie,oe,re=Object(X.b)(Object(g.a)((function(e){return{isBaseForumsActive:e.isActive({pathname:Object(P.m)()},!0),query:e.location.query,tabs:getTabs(e,te),updateLocation:e.push}})))((function(e){var t=e.tabs,n=e.updateLocation,s=e.children,i=Object(p.useState)(0),o=J()(i,2),r=o[0],a=o[1],c=Object(p.useState)(0),u=J()(c,2),l=u[0],m=u[1],b=Object(p.useState)(!1),f=J()(b,2),O=f[0],g=f[1],h=Object(p.useState)(!1),j=J()(h,2),y=j[0],v=j[1],x=t.activeTab,C=t.orderedTabs;return Object(p.useEffect)((function(){"foryou"===x&&a(r+1),"activity"===x&&m(l+1),r>=1&&g(!0),l>=1&&v(!0)}),[x]),Object(d.d)("section",null,Object(d.d)(se,{onEnter:function(e){return onTabEnter(n,e["data-link"])}},C.map((function(e){return Object(d.d)(DiscussionsLandingPageTabs_Tab,{key:e.tabKey,title:e.title,pathname:e.pathname,isActive:e.isActive,query:e.query,tabKey:e.tabKey})}))),Object(d.d)("div",null,s({activeTab:x,forYouHasBeenVisited:O,activityHasBeenVisited:y})))})),ae=t("u5mk");t("HjX4");var ce,ue,le,de,pe,me,be=Object(f.compose)(Object(P.k)({fields:["link","title","description","lastAnsweredAt","forumQuestionCount"]}),ae.a)((function(e){var t=e.courseForums.filter((function(e){return e.forumType.typeName===P.p.weekForumType})).map((function(t){var n=e.courseForumStatistics.find((function(e){return e.courseForumId===t.id}));return Object.assign({},t,{discussionsLink:v.a.join(Object(P.n)(e.courseSlug),t.link),lastAnsweredAt:n&&n.lastAnsweredAt,forumQuestionCount:n&&n.forumQuestionCount})})),n=Object(x.a)();return Object(d.d)(forumList_ForumsList,{className:"rc-DiscussionsWeekForums"},Object(d.d)("ul",{css:Object(d.c)(ie||(ie=r()(["\n          list-style: none;\n          margin: 0;\n          padding: 0;\n        "])))},t.map((function(e){return Object(d.d)("li",{key:e.id,css:Object(d.c)(oe||(oe=r()(["\n              border-bottom: 1px solid ",";\n              padding: ",";\n\n              &:hover {\n                background-color: ",";\n              }\n            "])),n.palette.gray[300],n.spacing(24),n.palette.blue[50])},Object(d.d)(forumList_ForumsLabel,{discussionsLink:e.discussionsLink,title:e.title,description:e.description,lastAnsweredAt:e.lastAnsweredAt,forumQuestionCount:e.forumQuestionCount}))}))))})),fe=(t("9M8W"),function(){var e=Object(x.a)();return Object(d.d)(h.a,{container:!0,item:!0,xs:12,role:"tabpanel"},ce||(ce=Object(d.d)(h.a,{item:!0,xs:12},Object(d.d)(U,null))),Object(d.d)(h.a,{item:!0,xs:12,css:Object(d.c)(ue||(ue=r()(["\n          margin: ",";\n        "])),e.spacing(0,0,24,0))},le||(le=Object(d.d)(V,null))),de||(de=Object(d.d)(h.a,{xs:12,item:!0},Object(d.d)(be,null))))}),Oe=function(e){function DiscussionsMainColumn(){return e.apply(this,arguments)||this}return l()(DiscussionsMainColumn,e),DiscussionsMainColumn.prototype.render=function(){var e,t=this.props,n=t.className,s=t.theme,i=Object(d.c)({".searchBoxContainer":c()({justifyContent:"flex-start",margin:s.spacing(0,16,0,0)},"".concat(s.breakpoints.up("xs")),{width:"330px"}),".newThreadButtonContainer":(e={},c()(e,"".concat(s.breakpoints.up("sm")),{padding:s.spacing(0)}),c()(e,"".concat(s.breakpoints.up("xs")),{width:"100px",minWidth:"100px"}),c()(e,"".concat(s.breakpoints.down("xs")),{padding:s.spacing(24,0,0,0)}),e)});return Object(d.d)(h.a,{container:!0,item:!0,xs:12,className:Y()("rc-DiscussionsMainColumn",n),css:i},Object(d.d)(h.a,{container:!0,item:!0,style:{display:"flex"}},Object(d.d)(h.a,{item:!0,xs:12,sm:5,className:"searchBoxContainer"},Object(d.d)(P.g,{previousElementId:"sortOrder",nextElementId:"new-thread-btn",placeholder:"Search forum",animateFromSearch:function(){return null},animateToSearch:function(){return null},onSubmit:function(){return null}})),pe||(pe=Object(d.d)(h.a,{item:!0,xs:12,sm:3,className:"newThreadButtonContainer"},Object(d.d)(P.d,null)))),Object(d.d)(h.a,{item:!0,xs:12,style:{padding:s.spacing(32,0,0,0)}},Object(d.d)(re,null,(function(e){var t=e.activeTab,n=e.forYouHasBeenVisited,s=e.activityHasBeenVisited;return Object(d.d)(p.Fragment,null,Object(d.d)(DiscussionsLandingPageTabs_TabPanel,{activeTab:t,tabName:te.FORYOU},Object(d.d)(P.e,{hasBeenVisited:n})),Object(d.d)(DiscussionsLandingPageTabs_TabPanel,{activeTab:t,tabName:te.ACTIVITY},Object(d.d)(P.i,{hasBeenVisited:s})),Object(d.d)(DiscussionsLandingPageTabs_TabPanel,{activeTab:t,tabName:te.ALL},me||(me=Object(d.d)(fe,null))))}))))},DiscussionsMainColumn}(p.Component);Oe.contextTypes={router:b.a.object.isRequired};var ge,he,je=Object(f.compose)(Object(g.a)((function(e){return{query:e.location.query.q&&decodeURIComponent(e.location.query.q)}})))(Object(y.a)(Oe)),ye=t("JOyW"),ve=function(e){function DiscussionsNotification(){for(var t,n=arguments.length,s=new Array(n),i=0;i<n;i++)s[i]=arguments[i];return(t=e.call.apply(e,[this].concat(s))||this).state={isVisible:!0},t}l()(DiscussionsNotification,e);var t=DiscussionsNotification.prototype;return t.handleDismiss=function(){this.setState({isVisible:!1})},t.render=function(){var e=this;return this.state.isVisible&&Object(d.d)("div",{className:"rc-WelcomeNotifications"},Object(d.d)(ye.a,{trackingName:"subscribed_to_digest_notification",type:"info",message:I()("You have been successfully subscribed to the digest."),isDismissible:!0,onDismiss:function(){return e.handleDismiss()}}))},DiscussionsNotification}(p.Component),xe=t("8cuT"),Ce=t("zXDh"),ke=t("fAYU"),Ae=t("KXbA"),Se=t("9A5E");var Ne,Te,we,DiscussionsDescription_DiscussionsGuidelines=function(){var e=Object(Ce.isRightToLeft)(I.a.getLocale())?"".concat(Ce.supportPageLocale.ar,"/"):"",t="https://learner.coursera.help/hc/".concat(e,"articles/208280036"),n=Object(x.a)();return Object(d.d)("div",{className:"rc-DiscussionsGuidelines",css:Object(d.c)(ge||(ge=r()(["\n        margin: ",";\n      "])),n.spacing(16,0,0,0))},Object(d.d)(ke.a,{variant:"quiet",typographyVariant:"body1",icon:he||(he=Object(d.d)(Ae.a,{size:"small"})),iconPosition:"after",component:Se.a,trackingName:"DiscussionsGuidelines",href:t,target:"_blank","aria-label":I()("Forum guidelines link opens in a new tab.")},I()("Forum guidelines")))};var Le=Object(f.compose)(Object(xe.a)(["CourseStore"],(function(e){var t=e.CourseStore;return{courseId:t.getCourseId(),courseSlug:t.getCourseSlug()}})),Object(P.k)({fields:["description","forumType"]}))((function(e){var t=Object(x.a)(),n=Object(d.d)(T.a,{className:"description"},I()("Welcome to the discussion forums!\n            Ask questions, debate ideas, and find classmates who share your goals.\n            Browse popular threads below or other forums in the sidebar."));if(e.courseForums&&e.courseForums.length){var s=e.courseForums.find((function(e){return e.forumType.typeName===P.p.rootForumType}));s&&!w.c.isEmpty(s.description)&&(n=Object(d.d)(L.a,{cml:s.description,isCdsEnabled:!0}))}return Object(d.d)(ContentBlock,{className:"rc-DiscussionsDescription",css:Object(d.c)(Ne||(Ne=r()(["\n        padding: ",";\n        margin: ",";\n      "])),t.spacing(24),t.spacing(0,0,24,0))},Object(d.d)(T.a,{variant:"h2semibold",css:Object(d.c)(Te||(Te=r()(["\n          margin: ",";\n        "])),t.spacing(0,0,16,0))},I()("Description")),n,we||(we=Object(d.d)(DiscussionsDescription_DiscussionsGuidelines,null)))})),De=t("jJ30"),Ie=t("qiY+"),Fe=function(e){function DiscussionsModerators(){return e.apply(this,arguments)||this}l()(DiscussionsModerators,e);var t=DiscussionsModerators.prototype;return t.componentWillMount=function(){var e=this.props.courseId;this.props.staffSocialProfiles.length||this.context.executeAction(De.b,{courseId:e})},t.render=function(){if(!this.props.staffSocialProfiles)return null;if(!this.props.staffSocialProfiles.find((function(e){return e.courseRole===Ie.a.MENTOR||e.courseRole===Ie.a.TEACHING_STAFF||e.courseRole===Ie.a.COURSE_ASSISTANT})))return null;var e=this.props.theme,t=Object(d.c)({padding:e.spacing(24),"ul.moderator-list li.staff-profile-container .rc-ProfileImage .c-profile-initials":G()({},e.typography.body1),".moderator-list":{display:"flex",flexFlow:"row",flexWrap:"wrap",padding:e.spacing(16,0,0,0),".staff-profile-container":{position:"relative",display:"inline-block",margin:e.spacing(0,16,12,0)}}});return Object(d.d)("div",{css:t,className:"rc-DiscussionsModerators card-no-action"},Object(d.d)(T.a,{variant:"h2semibold"},I()("Moderators")),Object(d.d)("ul",{className:"moderator-list"},this.props.staffSocialProfiles.map((function(e){return Object(d.d)("li",{className:"staff-profile-container",key:e.id},Object(d.d)(P.f,{courseRole:e.courseRole,externalUserId:e.externalUserId,fullName:e.fullName,helperStatus:e.helperStatus,profileImageUrl:e.photoUrl}))}))))},DiscussionsModerators}(p.Component);Fe.contextTypes={executeAction:b.a.func.isRequired};var qe,Pe,Re,Ve,Be,Ee=Object(f.compose)(Object(xe.a)(["CourseStore","ClassmatesProfileStore"],(function(e){var t=e.CourseStore,n=e.ClassmatesProfileStore;return{courseId:t.getCourseId(),staffSocialProfiles:n.getStaffProfiles()}})))(Object(y.a)(Fe)),We=function(e){function DiscussionsSideColumn(){return e.apply(this,arguments)||this}return l()(DiscussionsSideColumn,e),DiscussionsSideColumn.prototype.render=function(){var e=this.props.className;return Object(d.d)("div",{className:Y()("rc-DiscussionsSideColumn",e)},qe||(qe=Object(d.d)(P.h,null)),Pe||(Pe=Object(d.d)(j.a,{smDown:!0},Object(d.d)(P.c,null))),Re||(Re=Object(d.d)(Le,null)),Ve||(Ve=Object(d.d)(Ee,null)))},DiscussionsSideColumn}(p.Component),Ye=t("15pW"),Me=t("ZJgU"),He=t("q5zD"),Ue=(t("cK2b"),function(e){function LandingPageSearchResultsSummary(){for(var t,n=arguments.length,s=new Array(n),i=0;i<n;i++)s[i]=arguments[i];return(t=e.call.apply(e,[this].concat(s))||this).cancelSearchResults=function(){t.context.router.push({pathname:t.context.router.location.pathname,params:t.context.router.params,query:{}})},t}return l()(LandingPageSearchResultsSummary,e),LandingPageSearchResultsSummary.prototype.render=function(){var e=this.props,t=e.numResults,n=e.query,s=e.loadingState;return n?Object(d.d)(h.a,{container:!0,item:!0,xs:12,className:"rc-LandingPageSearchResultsSummary "},Object(d.d)(T.a,{className:"search-results","aria-live":"polite",id:"landing-page-search-results"},s===P.o.DONE?Object(d.d)(N.a,{message:I()("{numResults} {numResults, plural,\n              =1 {search result} =0 {0 search results} other {search results}} for {query}"),numResults:t,query:Object(d.d)("strong",null,n)}):Object(d.d)(N.b,{message:I()("Loading search results...")})),Object(d.d)(Me.a,{type:"button",onClick:this.cancelSearchResults,variant:"ghost","data-e2e":"cancel-search-button"},Be||(Be=Object(d.d)(He.a,{name:"close",className:"color-secondary-text"})))):null},LandingPageSearchResultsSummary}(p.Component));Ue.propTypes={id:b.a.string,query:b.a.string,numResults:b.a.number,loadingState:b.a.oneOf(Object.keys(P.o))},Ue.contextTypes={router:b.a.object.isRequired};var Qe=Object(f.compose)(Object(g.a)((function(e){return{query:e.location.query.q&&decodeURIComponent(e.location.query.q)}})),Object(xe.a)(["DiscussionsSearchStore"],(function(e,t){var n=e.DiscussionsSearchStore;return{numResults:n.getNumResults({forumId:t.id,query:t.query}),loadingState:n.loadingState}})))(Ue),Je=t("BZ+2"),Ke=(t("tmK+"),function(e){function LandingPageThreadsViewWrapper(){for(var t,n=arguments.length,s=new Array(n),i=0;i<n;i++)s[i]=arguments[i];return(t=e.call.apply(e,[this].concat(s))||this).styles=function(){return{width:"100%","div.rc-LandingPageThreadsViewWrapper":{maxWidth:"unset","div.forum-info.caption-text.color-secondary-text":G()(G()({},t.props.theme.typography.body1),{},{position:"relative",left:"-33px"}),"div.headline-1-text.question-title.color-primary-text.question-title-bold":G()({},t.props.theme.typography.h3bold),"& .rc-ThreadBadge.bgcolor-accent-brown-light":G()(G()({},t.props.theme.typography.h4bold),{},{backgroundColor:t.props.theme.palette.purple[50],color:t.props.theme.palette.purple[700]}),"& li.rc-ThreadsListEntry .cif-stack.pin-container":{height:"24px",width:"24px",margin:t.props.theme.spacing(0,8,0,0),position:"relative",top:"24px","& .cif-circle.cif-stack-2x.pin-background":{"&:before":{content:"none"},backgroundColor:t.props.theme.palette.yellow[500],borderRadius:"100%",height:"24px",width:"24px"},"& .cif-pin.cif-stack-lg.pin-icon":{margin:0,position:"relative",top:"-7px"}},"& .rc-Metadata.entry-metadata.caption-text.color-secondary-text":G()({},t.props.theme.typography.body2),"& .rc-ProfileName.nostyle.pii-hide":G()(G()({},t.props.theme.typography.h4bold),{},{color:t.props.theme.palette.blue[600]}),"& .ThreadListEntry button.c-new-thread-button.secondary":G()({},t.props.theme),".rc-DiscussionsBody, .rc-DiscussionsSearchResults":{width:"100%"},".rc-LandingPageSearchResultsSummary":{display:"flex",justifyContent:"space-between",alignItems:"center",backgroundColor:t.props.theme.palette.blue[50]}}}},t}return l()(LandingPageThreadsViewWrapper,e),LandingPageThreadsViewWrapper.prototype.render=function(){if(!this.props.currentForum)return null;var e=v.a.join(Ye.d.learnRoot,this.props.courseSlug,this.props.currentForum.link);return Object(d.d)(h.a,{container:!0,item:!0,xs:12,className:"rc-LandingPageThreadsViewWrapper"},Object(d.d)(Qe,{id:this.props.currentForum.forumId}),Object(d.d)(P.b,{backLink:e}))},LandingPageThreadsViewWrapper}(p.Component));Ke.propTypes={search:b.a.string,currentForum:b.a.instanceOf(Je.a),courseSlug:b.a.string};var Ge,Xe,_e,ze,Ze,$e=Object(f.compose)(Object(P.k)({fields:["link"]}),Object(g.a)(P.q))(Ke);t.d(e,"DiscussionsLandingPage",(function(){return et}));var et=function(e){function DiscussionsLandingPage(){for(var t,n=arguments.length,s=new Array(n),i=0;i<n;i++)s[i]=arguments[i];return(t=e.call.apply(e,[this].concat(s))||this).layoutCss={},t.layoutStyle=function(e){var n;return n={},c()(n,t.props.theme.breakpoints.up("sm"),{".mainColumn .rc-DiscussionsLandingPage":{margin:0}}),c()(n,t.props.theme.breakpoints.down("sm"),{".mainColumn":{padding:e.spacing(0),margin:e.spacing(0,0,48,0)}}),c()(n,".columnContainer",{paddingTop:t.props.theme.spacing(0)}),n},t}l()(DiscussionsLandingPage,e);var t=DiscussionsLandingPage.prototype;return t.getChildContext=function(){return{courseId:this.props.courseId,userId:this.props.userId.toString()}},t.componentDidMount=function(){var e=this.props,t=e.addToHelper,n=e.userId,s=e.courseId,i=e.maybeAddHelpers;t&&i({variables:{helperId:"".concat(n,"~").concat(s)}})},t.render=function(){var e=this.props.addToHelper;return Object(d.d)(h.a,{css:this.layoutStyle(this.props.theme),container:!0,item:!0,xs:12,className:"rc-DiscussionsLandingPage",spacing:24},e&&(Ge||(Ge=Object(d.d)(h.a,{item:!0,xs:12},Object(d.d)(ve,null)))),Xe||(Xe=Object(d.d)(h.a,{item:!0,xs:12},Object(d.d)(personalization_DiscussionsLandingPageHeader,null))),Object(d.d)(j.a,{mdUp:!0,css:Object(d.c)(_e||(_e=r()(["\n            margin: ",";\n            width: 100%;\n          "])),this.props.theme.spacing(0,12,0,12))},ze||(ze=Object(d.d)(P.c,null))),Ze||(Ze=Object(d.d)(h.a,{className:"columnContainer",container:!0,item:!0,xs:12,justifyContent:"space-between"},Object(d.d)(h.a,{item:!0,className:"mainColumn",xs:12,md:9,spacing:32},Object(d.d)(je,null)),Object(d.d)(h.a,{item:!0,xs:12,md:3},Object(d.d)(We,null)))))},DiscussionsLandingPage}(p.Component);et.childContextTypes={courseId:b.a.string,userId:b.a.string};e.default=Object(f.compose)(Object(P.k)({fields:[],subcomponents:[V,be,$e]}),Object(g.a)((function(e){return{addToHelper:"true"===e.location.query.addToHelper}})),P.r)(Object(y.a)(et))},"4vdf":function(module,exports,e){},"9M8W":function(module,exports,e){e("+tS/")},AEB4:function(module,exports,e){e("ftHD")},HjX4:function(module,exports,e){e("Jg71")},HkWL:function(module,exports,e){e("4vdf")},JBNF:function(module,exports,e){e("lio+")},JOyW:function(module,e,t){"use strict";var n=t("pVnL"),s=t.n(n),i=t("VbXa"),o=t.n(i),r=t("q1tI"),a=t("IqPN"),c=t("sOkY"),u=(t("AEB4"),function(e){function InContextNotification(){return e.apply(this,arguments)||this}return o()(InContextNotification,e),InContextNotification.prototype.render=function(){var e=this.props.trackingName;return(r.createElement(c.a,{trackClicks:!1,withVisibilityTracking:!0,trackingName:e,className:"rc-InContextNotification body-1-text"},r.createElement(a.a,s()({},this.props,{htmlAttributes:{"data-classname":"in-context-notification"}}))))},InContextNotification}(r.Component));e.a=u},Jg71:function(module,exports,e){},O2fD:function(module,exports,e){},a9ka:function(module,exports,e){e("rnpO")},cK2b:function(module,exports,e){e("sX3l")},ftHD:function(module,exports,e){},"lio+":function(module,exports,e){},rnpO:function(module,exports,e){},sX3l:function(module,exports,e){},"tmK+":function(module,exports,e){e("O2fD")}}]);
//# sourceMappingURL=63.4e0d7a956218e4fd5240.js.map