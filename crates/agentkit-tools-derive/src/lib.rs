//! Procedural macros for declaring agentkit tools.
//!
//! Currently exposes one attribute macro, [`macro@tool`], that turns an
//! async function into a unit struct implementing
//! `agentkit_tools_core::Tool`. The struct's name matches the function's
//! identifier, so registering it reads as `registry.with(my_tool)`.
//!
//! The companion crate `agentkit-tools-core` provides the `tool_spec_for`
//! runtime helper that the generated code calls to derive the JSON Schema
//! from the input type via `schemars`.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    Attribute, Expr, ExprLit, FnArg, ItemFn, Lit, Meta, MetaNameValue, Pat, PatType, ReturnType,
    Token, Type, parse_macro_input, punctuated::Punctuated,
};

/// `#[tool]` attribute. Apply to an async function whose first argument is
/// a deserializable input struct that implements `schemars::JsonSchema`.
///
/// Recognised arguments:
///
/// - `name = "literal"` — overrides the tool's name. Defaults to the
///   function's identifier.
/// - `description = "literal"` — sets the tool's description. Defaults to
///   the function's first doc comment line if present, otherwise an empty
///   string.
///
/// The generated struct shadows the function name and is `Default` +
/// `Clone`-able, so it slots straight into a `ToolRegistry`.
///
/// # Example
///
/// ```rust,ignore
/// use agentkit_tools_core::{ToolError, ToolResult};
/// use agentkit_tools_derive::tool;
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// #[derive(JsonSchema, Deserialize)]
/// struct Input { city: String }
///
/// /// Fetch weather for a city.
/// #[tool]
/// async fn get_weather(input: Input) -> Result<ToolResult, ToolError> {
///     // ...
///     # unimplemented!()
/// }
/// ```
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = parse_macro_input!(attr with Punctuated::<Meta, Token![,]>::parse_terminated);

    match expand_tool(attrs.into_iter().collect(), func) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

fn expand_tool(args: Vec<Meta>, func: ItemFn) -> syn::Result<TokenStream2> {
    let fn_name = func.sig.ident.clone();
    let fn_name_string = fn_name.to_string();

    if func.sig.asyncness.is_none() {
        return Err(syn::Error::new_spanned(
            &func.sig,
            "#[tool] requires an async function",
        ));
    }

    let input_type = extract_input_type(&func)?;

    let mut name_override: Option<String> = None;
    let mut description_override: Option<String> = None;

    for meta in args {
        match meta {
            Meta::NameValue(MetaNameValue {
                path,
                value:
                    Expr::Lit(ExprLit {
                        lit: Lit::Str(s), ..
                    }),
                ..
            }) => {
                if path.is_ident("name") {
                    name_override = Some(s.value());
                } else if path.is_ident("description") {
                    description_override = Some(s.value());
                } else {
                    return Err(syn::Error::new_spanned(
                        path,
                        "unknown #[tool] argument; expected `name` or `description`",
                    ));
                }
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "expected `name = \"...\"` or `description = \"...\"`",
                ));
            }
        }
    }

    let tool_name = name_override.unwrap_or_else(|| fn_name_string.clone());
    let description = description_override
        .unwrap_or_else(|| extract_doc_comment(&func.attrs).unwrap_or_default());

    let body = &func.block;
    let input_arg_ident = first_arg_pat(&func)?;
    let user_return_type = match &func.sig.output {
        ReturnType::Type(_, ty) => quote! { #ty },
        ReturnType::Default => {
            return Err(syn::Error::new_spanned(
                &func.sig,
                "#[tool] requires an explicit return type, e.g. `-> Result<ToolResult, ToolError>`",
            ));
        }
    };

    // The user's body returns `Result<ToolResult, ToolError>`. The macro
    // overwrites the result's `call_id` with the inbound `request.call_id`
    // so users can build results with `Default::default()` or any
    // placeholder; the model's call_id is what ends up in the transcript.
    let expanded = quote! {
        #[allow(non_camel_case_types)]
        #[derive(Default, Clone)]
        pub struct #fn_name;

        #[::agentkit_tools_core::__private_async_trait::async_trait]
        impl ::agentkit_tools_core::Tool for #fn_name {
            fn spec(&self) -> &::agentkit_tools_core::ToolSpec {
                static SPEC: ::std::sync::OnceLock<::agentkit_tools_core::ToolSpec> =
                    ::std::sync::OnceLock::new();
                SPEC.get_or_init(|| {
                    ::agentkit_tools_core::tool_spec_for::<#input_type>(
                        #tool_name,
                        #description,
                    )
                })
            }

            async fn invoke(
                &self,
                request: ::agentkit_tools_core::ToolRequest,
                _ctx: &mut ::agentkit_tools_core::ToolContext<'_>,
            ) -> ::std::result::Result<
                ::agentkit_tools_core::ToolResult,
                ::agentkit_tools_core::ToolError,
            > {
                let __call_id = request.call_id.clone();
                let #input_arg_ident: #input_type = ::serde_json::from_value(request.input)
                    .map_err(|e| ::agentkit_tools_core::ToolError::InvalidInput(e.to_string()))?;
                let __body_fut = async move { let __body_out: #user_return_type = #body; __body_out };
                let mut __result: ::agentkit_tools_core::ToolResult = __body_fut.await?;
                __result.result.call_id = __call_id;
                ::std::result::Result::Ok(__result)
            }
        }
    };

    Ok(expanded)
}

fn extract_input_type(func: &ItemFn) -> syn::Result<Type> {
    let first = func.sig.inputs.first().ok_or_else(|| {
        syn::Error::new_spanned(
            &func.sig,
            "#[tool] requires at least one argument: the input type",
        )
    })?;
    match first {
        FnArg::Typed(PatType { ty, .. }) => Ok((**ty).clone()),
        FnArg::Receiver(_) => Err(syn::Error::new_spanned(
            first,
            "#[tool] cannot be applied to a method (no `self` allowed)",
        )),
    }
}

fn first_arg_pat(func: &ItemFn) -> syn::Result<TokenStream2> {
    let first =
        func.sig.inputs.first().ok_or_else(|| {
            syn::Error::new_spanned(&func.sig, "#[tool] requires an input argument")
        })?;
    if let FnArg::Typed(PatType { pat, .. }) = first {
        if let Pat::Ident(ident) = pat.as_ref() {
            let id = &ident.ident;
            return Ok(quote! { #id });
        }
        return Err(syn::Error::new_spanned(
            pat,
            "#[tool] requires the input argument to be a simple identifier pattern (e.g. `input: MyInput`)",
        ));
    }
    Err(syn::Error::new_spanned(first, "unsupported argument shape"))
}

fn extract_doc_comment(attrs: &[Attribute]) -> Option<String> {
    let mut buf = String::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let Meta::NameValue(MetaNameValue {
            value: Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }),
            ..
        }) = &attr.meta
        {
            let line = s.value();
            let trimmed = line.trim();
            if trimmed.is_empty() {
                if !buf.is_empty() {
                    break;
                }
                continue;
            }
            if !buf.is_empty() {
                buf.push(' ');
            }
            buf.push_str(trimmed);
        }
    }
    if buf.is_empty() { None } else { Some(buf) }
}
