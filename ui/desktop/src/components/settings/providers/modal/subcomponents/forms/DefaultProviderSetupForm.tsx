import React, { useEffect, useMemo, useState, useCallback } from 'react';
import { Input } from '../../../../../ui/input';
import { useConfig } from '../../../../../ConfigContext'; // Adjust this import path as needed
import { ProviderDetails, ConfigKey } from '../../../../../../api';

interface ValidationErrors {
  [key: string]: string;
}

interface DefaultProviderSetupFormProps {
  configValues: Record<string, string>;
  setConfigValues: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  provider: ProviderDetails;
  validationErrors: ValidationErrors;
}

export default function DefaultProviderSetupForm({
  configValues,
  setConfigValues,
  provider,
  validationErrors = {},
}: DefaultProviderSetupFormProps) {
  const parameters = useMemo(
    () => provider.metadata.config_keys || [],
    [provider.metadata.config_keys]
  );
  const [isLoading, setIsLoading] = useState(true);
  const { read } = useConfig();

  console.log('configValues default form', configValues);

  // Initialize values when the component mounts or provider changes
  const loadConfigValues = useCallback(async () => {
    setIsLoading(true);
    const newValues = { ...configValues };

    // Try to load actual values from config for each parameter that is not secret
    for (const parameter of parameters) {
      if (parameter.required) {
        try {
          // Check if there's a stored value in the config system
          const configKey = `${parameter.name}`;
          const configResponse = await read(configKey, parameter.secret || false);

          if (configResponse) {
            // Use the value from the config provider
            newValues[parameter.name] = String(configResponse);
          } else if (
            parameter.default !== undefined &&
            parameter.default !== null &&
            !configValues[parameter.name]
          ) {
            // Fall back to default value if no config value exists
            newValues[parameter.name] = String(parameter.default);
          }
        } catch (error) {
          console.error(`Failed to load config for ${parameter.name}:`, error);
          // Fall back to default if read operation fails
          if (
            parameter.default !== undefined &&
            parameter.default !== null &&
            !configValues[parameter.name]
          ) {
            newValues[parameter.name] = String(parameter.default);
          }
        }
      }
    }

    // Update state with loaded values
    setConfigValues((prev) => ({
      ...prev,
      ...newValues,
    }));
    setIsLoading(false);
  }, [configValues, parameters, read, setConfigValues]);

  useEffect(() => {
    loadConfigValues();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Helper function to generate appropriate placeholder text
  const getPlaceholder = (parameter: ConfigKey): string => {
    // If default is defined and not null, show it
    if (parameter.default !== undefined && parameter.default !== null) {
      return `Default: ${parameter.default}`;
    }

    // Otherwise, use the parameter name as a hint
    return parameter.name.toUpperCase();
  };

  if (isLoading) {
    return <div className="text-center py-4">Loading configuration values...</div>;
  }

  // Use all parameters for rendering, not just required ones.
  // The 'required' property on the parameter can be used for form validation if needed,
  // but all configurable parameters should be displayed.
  return (
    <div className="mt-4 space-y-4">
      {parameters.length === 0 ? (
        <div className="text-center text-gray-500">
          No configuration available for this provider.
        </div>
      ) : (
        parameters.map((parameter) => (
          <div key={parameter.name}>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {parameter.name
                .split('_')
                .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
                .join(' ')}
              {parameter.required && <span className="text-red-500 ml-1">*</span>}
            </label>
            <Input
              type={parameter.secret ? 'password' : 'text'}
              value={configValues[parameter.name] || ''}
              onChange={(e) =>
                setConfigValues((prev) => ({
                  ...prev,
                  [parameter.name]: e.target.value,
                }))
              }
              placeholder={getPlaceholder(parameter)}
              className={`w-full h-14 px-4 font-regular rounded-lg shadow-none ${
                validationErrors[parameter.name]
                  ? 'border-2 border-red-500'
                  : 'border border-gray-300'
              } bg-white text-lg placeholder:text-gray-400 font-regular text-gray-900`}
              required={parameter.required} // Set required attribute based on parameter
            />
            {validationErrors[parameter.name] && (
              <p className="text-sm text-red-500 mt-1">{validationErrors[parameter.name]}</p>
            )}
          </div>
        ))
      )}
    </div>
  );
}
