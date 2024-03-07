function main(input) {
    // Function to identify houses not marked with vowels
    function identifyNonRenovatedHouses(houses) {
        // Define a string of vowels, including uppercase for case-insensitive comparison
        const vowels = 'aeiouAEIOU';
        let nonRenovatedHouses = '';

        // Iterate through each character in the input string
        for (let i = 0; i < houses.length; i++) {
            let house = houses[i];
            // If the character is not a vowel, add it to the result string
            if (!vowels.includes(house)) {
                nonRenovatedHouses += house;
            }
        }

        return nonRenovatedHouses;
    }

    // Trim the input to remove any leading/trailing whitespace and identify non-renovated houses
    const nonRenovatedHouses = identifyNonRenovatedHouses(input.trim());

    // If there are any non-renovated houses, write them to STDOUT; otherwise, do not print anything
    if (nonRenovatedHouses) {
        process.stdout.write(nonRenovatedHouses);
    }
}

// Standard input processing setup
process.stdin.resume();
process.stdin.setEncoding("utf-8");
var stdin_input = "";

process.stdin.on("data", function (input) {
    stdin_input += input; // Collect input
});

process.stdin.on("end", function () {
    main(stdin_input); // Execute the main function once all input is received
});