package acpagent

import (
	"fmt"

	acp "github.com/caelis-labs/acp-go-sdk"

	"github.com/kirahowe/agentloop"
)

// The model a session runs, as ACP shows it: one session config option in
// the "model" category, a select with a single value. A client can show
// it; it cannot change it.

// ModelConfigID is the id of the model's config option.
const ModelConfigID acp.SessionConfigId = "model"

// Described is what the server knows of its model: the request's model id
// and whatever the provider said about it. The id stays the request's,
// since that is what every call sends; the provider's display name and
// window are taken when it gave them. A negative window means unknown.
func Described(id string, got agentloop.ModelInfo) agentloop.ModelInfo {
	return agentloop.ModelInfo{ID: id, DisplayName: got.DisplayName, ContextWindow: max(got.ContextWindow, 0)}
}

// ConfigOptions are the session config options for a model: the model as
// a select whose only value is its id, named by its display name, or by
// its id when the provider gave none. With no id there are none.
func ConfigOptions(info agentloop.ModelInfo) []acp.SessionConfigOption {
	if info.ID == "" {
		return nil
	}
	name := info.DisplayName
	if name == "" {
		name = info.ID
	}
	value := acp.SessionConfigValueId(info.ID)
	values := acp.SessionConfigSelectOptionsUngrouped{{Value: value, Name: name}}
	opt := acp.NewSessionConfigOptionSelect(value, acp.SessionConfigSelectOptions{Ungrouped: &values})
	category := acp.SessionConfigOptionCategoryModel
	opt.Select.Id = ModelConfigID
	opt.Select.Name = "Model"
	opt.Select.Category = &category
	return []acp.SessionConfigOption{opt}
}

// SetConfigOption answers session/set_config_option. The only change it
// accepts is setting the model to the one it already is, which answers
// with the options as they are. Anything else is refused as invalid
// params: the model is fixed by the request.
func SetConfigOption(info agentloop.ModelInfo, req acp.SetSessionConfigOptionRequest) (acp.SetSessionConfigOptionResponse, *acp.RequestError) {
	opts := ConfigOptions(info)
	refuse := func(msg string, configID acp.SessionConfigId) (acp.SetSessionConfigOptionResponse, *acp.RequestError) {
		return acp.SetSessionConfigOptionResponse{}, &acp.RequestError{Code: -32602, Message: msg, Data: map[string]any{"configId": string(configID)}}
	}
	if req.Boolean != nil {
		return refuse(fmt.Sprintf("agentloop has no boolean option %q", req.Boolean.ConfigId), req.Boolean.ConfigId)
	}
	v := req.ValueId
	if v == nil || v.ConfigId != ModelConfigID || len(opts) == 0 {
		var id acp.SessionConfigId
		if v != nil {
			id = v.ConfigId
		}
		return refuse(fmt.Sprintf("agentloop has no config option %q", id), id)
	}
	if string(v.Value) != info.ID {
		return refuse(fmt.Sprintf("the model is fixed at %s by agentloop's request; switching to %s is not supported", info.ID, v.Value), v.ConfigId)
	}
	return acp.SetSessionConfigOptionResponse{ConfigOptions: opts}, nil
}
