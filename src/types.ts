export interface OutputProps {
  addToOutput: (message: string, isError?: boolean) => void;
}

export interface OutputSectionProps {
  output: string;
  onClear: () => void;
}